from typing import Optional

import torch
from einops import rearrange, repeat
from torch import nn

from sige.nn import Gather, Scatter, SIGEConv2d, SIGEModule
from .attention import CrossAttention, default, exists, FeedForward, Normalize, SpatialTransformer, zero_module
from .diffusionmodules.sige_model import my_group_norm


class SIGECrossAttention(SIGEModule):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** (-0.5)
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

        self.cached_k = None
        self.cached_v = None

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        if self.mode == "full":
            k = self.to_k(context)
            v = self.to_v(context)
            self.cached_k = k
            self.cached_v = v
        else:
            k = self.cached_k
            v = self.cached_v
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        sim = torch.bmm(q, k.permute(0, 2, 1)) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        # out = einsum("b i j, b j d -> b i d", attn, v)
        out = torch.bmm(attn, v)

        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class SIGEBasicTransformerBlock(SIGEModule):
    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True, use_checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = SIGECrossAttention(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, full_x=None, context=None):
        x = self.attn1(self.norm1(x), context=None if full_x is None else self.norm1(full_x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SIGESpatialTransformer(SIGEModule, SpatialTransformer):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        use_checkpoint=True,
        block_size: Optional[int] = 4,
    ):
        super(SpatialTransformer, self).__init__()
        SIGEModule.__init__(self, call_super=False)
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        support_sparse = block_size is not None
        Conv2d = SIGEConv2d if support_sparse else nn.Conv2d

        self.support_sparse = support_sparse

        self.proj_in = Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                SIGEBasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    use_checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )

        self.proj_out = zero_module(Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))

        if support_sparse:
            self.gather = Gather(self.proj_in, block_size)
            self.scatter1 = Scatter(self.gather)
            self.scatter2 = Scatter(self.gather)
        self.scale, self.shift = None, None

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x

        if self.mode == "full":
            if self.support_sparse:
                x = self.gather(x)
            x, scale, shift = my_group_norm(x, self.norm)
            self.scale, self.shift = scale, shift
        elif self.mode in ("sparse", "profile"):
            if self.support_sparse:
                x = self.gather(x, self.scale, self.shift)
            else:
                x = x * self.scale + self.shift
        else:
            raise NotImplementedError("Unknown mode [%s]!!!" % self.mode)

        x = self.proj_in(x)

        if self.support_sparse:
            full_x = self.scatter1(x)
            full_x = rearrange(full_x, "b c h w -> b (h w) c")
            if self.mode == "full":
                x = full_x
            else:
                cc = x.size(1)
                x = x.view(b, -1, cc, x.size(2) * x.size(3))  # [b, nb, c, bh * bw]
                x = x.transpose(2, 3).reshape(b, -1, cc)
        else:
            full_x = None
            x = rearrange(x, "b c h w -> b (h w) c")

        for block in self.transformer_blocks:
            x = block(x, full_x=full_x, context=context)
        if self.support_sparse:
            # [b, nb * bh * bw, c] -> [b * nb, c, bh, bw]
            if self.mode == "full":
                x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            else:
                cc = x.size(-1)
                # [b, nb * bh * bw, c] -> [b, nb, bh * bw, c]
                x = x.view(b, -1, self.gather.block_size[0] * self.gather.block_size[1], cc)  # [b, nb, bh * bw, c]
                x = x.permute(0, 1, 3, 2).view(-1, cc, self.gather.block_size[0], self.gather.block_size[1])
        else:
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        x = self.proj_out(x)
        if self.support_sparse:
            x = self.scatter2(x, x_in)
        else:
            x = x + x_in
        return x
