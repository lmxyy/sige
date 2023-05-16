import functools
import math

import torch
from torch import nn
from torch.nn import functional as F

from sige.nn import Gather, Scatter, ScatterGather, ScatterWithBlockResidual, SIGEConv2d, SIGEModel, SIGEModule
from ..common import get_timestep_embedding, my_group_norm, Normalize, swish


class SIGEResnetBlock(SIGEModule):
    def __init__(
        self,
        args,
        config,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        temb_channels=512,
        resample=None,
        support_sparse=False,
    ):
        super(SIGEResnetBlock, self).__init__()
        self.args = args
        self.config = config
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.temb_channels = temb_channels
        self.resample = resample
        self.support_sparse = support_sparse

        main_block_size = config.model.sige_block_size.normal
        main_support_sparse = support_sparse and main_block_size is not None
        MainConv2d = SIGEConv2d if main_support_sparse else nn.Conv2d

        if resample == "down":
            self.pooling = nn.AvgPool2d(2)
            self.resample_func = self.pooling
        elif resample == "up":
            self.resample_func = functools.partial(F.interpolate, scale_factor=2)
        else:
            self.resample_func = lambda x: x

        self.norm1 = Normalize(in_channels)
        self.conv1 = MainConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels * 2)
        self.norm2 = Normalize(out_channels)
        self.conv2 = MainConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if main_support_sparse:
            self.main_gather = Gather(
                self.conv1, block_size=main_block_size, activation_name="swish" if resample is None else "identity"
            )
            self.scatter_gather = ScatterGather(self.main_gather, activation_name="swish")

        if self.in_channels != self.out_channels:
            shortcut_block_size = config.model.sige_block_size.instance
            shortcut_support_sparse = main_support_sparse and shortcut_block_size is not None
            ShortcutConv2d = SIGEConv2d if shortcut_support_sparse else nn.Conv2d
            self.nin_shortcut = ShortcutConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if shortcut_support_sparse:
                self.shortcut_gather = Gather(self.nin_shortcut, shortcut_block_size)
                self.scatter = ScatterWithBlockResidual(self.main_gather, self.shortcut_gather)
            elif main_support_sparse:
                self.scatter = Scatter(self.main_gather)
        else:
            if main_support_sparse:
                self.scatter = Scatter(self.main_gather)
            shortcut_support_sparse = False

        self.main_support_sparse = main_support_sparse
        self.shortcut_support_sparse = shortcut_support_sparse

        self.scale1s, self.shift1s = {}, {}
        self.scale2s, self.shift2s = {}, {}

    def forward(self, x, temb):
        if self.mode == "full":
            return self.full_forward(x, temb)
        elif self.mode in ["sparse", "profile"]:
            return self.sparse_forward(x)
        else:
            raise NotImplementedError("Unknown mode [%s]!!!" % self.mode)

    def full_forward(self, x, temb):
        cache_id = self.cache_id
        main_support_sparse = self.main_support_sparse
        shortcut_support_sparse = self.shortcut_support_sparse

        h = x
        x = self.resample_func(x)
        if self.in_channels != self.out_channels:
            if shortcut_support_sparse:
                x = self.shortcut_gather(x)
            x = self.nin_shortcut(x)

        h, scale, shift = my_group_norm(h, self.norm1)
        self.scale1s[cache_id], self.shift1s[cache_id] = scale, shift
        h = swish(h)
        h = self.resample_func(h)

        if main_support_sparse:
            h = self.main_gather(h)

        h = self.conv1(h)

        if main_support_sparse:
            h = self.scatter_gather(h)

        h, scale, shift = my_group_norm(h, self.norm2)
        emb_out = self.temb_proj(swish(temb))
        emb_scale, emb_shift = emb_out[:, : self.out_channels], emb_out[:, self.out_channels :]
        h = h * (1 + emb_scale.view(1, -1, 1, 1)) + emb_shift.view(1, -1, 1, 1)
        scale = (1 + emb_scale[0]) * scale
        shift = (1 + emb_scale[0]) * shift
        shift = shift + emb_shift[0]
        self.scale2s[cache_id], self.shift2s[cache_id] = scale, shift

        h = swish(h)
        h = self.conv2(h)

        if main_support_sparse:
            h = self.scatter(h, x)
        else:
            h = h + x
        return h

    def sparse_forward(self, x):
        cache_id = self.cache_id
        main_support_sparse = self.main_support_sparse
        shortcut_support_sparse = self.shortcut_support_sparse

        h = x
        x = self.resample_func(x)
        if self.in_channels != self.out_channels:
            if shortcut_support_sparse:
                x = self.shortcut_gather(x)
            x = self.nin_shortcut(x)

        if main_support_sparse:
            if self.resample is None:
                h = self.main_gather(
                    h, self.scale1s[cache_id].view(1, -1, 1, 1), self.shift1s[cache_id].view(1, -1, 1, 1)
                )
            else:
                h = h * self.scale1s[cache_id].view(1, -1, 1, 1) + self.shift1s[cache_id].view(1, -1, 1, 1)
                h = swish(h)
                h = self.resample_func(h)
                h = self.main_gather(h)
        else:
            h = h * self.scale1s[cache_id].view(1, -1, 1, 1) + self.shift1s[cache_id].view(1, -1, 1, 1)
            h = swish(h)
            h = self.resample_func(h)

        h = self.conv1(h)

        if main_support_sparse:
            h = self.scatter_gather(
                h, self.scale2s[cache_id].view(1, -1, 1, 1), self.shift2s[cache_id].view(1, -1, 1, 1)
            )
        else:
            h = h * self.scale2s[cache_id].view(1, -1, 1, 1) + self.shift2s[cache_id].view(1, -1, 1, 1)
            h = swish(h)

        h = self.conv2(h)

        if main_support_sparse:
            h = self.scatter(h, x)
        else:
            h = h + x
        return h


class SIGEAttnBlock(SIGEModule):
    def __init__(self, args, config, in_channels, head_dim=None, num_heads=None, support_sparse=False):
        super(SIGEAttnBlock, self).__init__()

        if head_dim is None:
            assert num_heads is not None
            assert in_channels % num_heads == 0
            head_dim = in_channels // num_heads
        else:
            assert num_heads is None
            assert in_channels % head_dim == 0
            num_heads = in_channels // head_dim

        self.args = args
        self.config = config
        self.in_channels = in_channels
        self.num_heads, self.head_dim = num_heads, head_dim

        block_size = config.model.sige_block_size.instance
        support_sparse = support_sparse and block_size is not None
        self.support_sparse = support_sparse

        self.norm = Normalize(in_channels)
        Conv2d = SIGEConv2d if support_sparse else nn.Conv2d
        self.qkv = Conv2d(in_channels, num_heads * head_dim * 3, kernel_size=1, stride=1, padding=0)
        self.proj_out = Conv2d(num_heads * head_dim, in_channels, kernel_size=1, stride=1, padding=0)

        if support_sparse:
            self.gather1 = Gather(self.qkv, block_size=block_size)
            self.scatter1 = Scatter(self.gather1)
            self.gather2 = Gather(self.proj_out, block_size=block_size)
            self.scatter2 = Scatter(self.gather2)

        self.scales, self.shifts = {}, {}

    def clear_cache(self):
        self.scales, self.shifts = {}, {}

    def forward(self, x):
        cache_id = self.cache_id
        h_ = x

        if self.mode == "full":
            if self.support_sparse:
                h_ = self.gather1(h_)  # record the input resolution
            h_, scale, shift = my_group_norm(h_, self.norm)
            self.scales[cache_id], self.shifts[cache_id] = scale, shift
        elif self.mode in ["sparse", "profile"]:
            if self.support_sparse:
                h_ = self.gather1(h_, self.scales[cache_id].view(1, -1, 1, 1), self.shifts[cache_id].view(1, -1, 1, 1))
            else:
                h_ = h_ * self.scales[cache_id].view(1, -1, 1, 1) + self.shifts[cache_id].view(1, -1, 1, 1)
        else:
            raise NotImplementedError("Unknown mode [%s]!!!" % self.mode)

        qkv = self.qkv(h_)
        if self.support_sparse:
            qkv = self.scatter1(qkv)

        num_heads, head_dim = self.num_heads, self.head_dim
        c = num_heads * head_dim
        q = qkv[:, :c]
        k = qkv[:, c : 2 * c]
        v = qkv[:, 2 * c :]

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b * num_heads, head_dim, h * w)
        q = q.permute(0, 2, 1)  # b * nh, hw, hd
        k = k.reshape(b * num_heads, head_dim, h * w)  # b * nh, hd, hw
        w_ = torch.bmm(q, k)  # b * nh, hw, hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (head_dim ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b * num_heads, head_dim, h * w)
        w_ = w_.permute(0, 2, 1)  # b*nh, hw, hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)  # [b*nb, hd, hw]
        h_ = h_.view(b, c, h, w)

        if self.support_sparse:
            h_ = self.gather2(h_)
        h_ = self.proj_out(h_)

        if self.support_sparse:
            h_ = self.scatter2(h_, x)
        else:
            h_ = h_ + x
        return h_


class SIGEUNet(SIGEModel):
    def __init__(self, args, config):
        super(SIGEUNet, self).__init__()
        self.args = args
        self.config = config
        self.support_sparse = False

        ch, ch_mult = config.model.ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        in_ch, out_ch = config.model.in_ch, config.model.out_ch
        resolution = config.data.image_size
        num_heads = config.model.num_heads
        head_dim = config.model.head_dim
        sparse_resolution_threshold = config.model.sparse_resolution_threshold

        self.ch = ch
        self.temb_ch = config.model.temb_ch
        self.logsnr_input_type = config.model.logsnr_input_type

        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.num_heads = num_heads
        self.head_dim = head_dim

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList(
            [
                nn.Linear(self.ch, self.temb_ch),
                nn.Linear(self.temb_ch, self.temb_ch),
            ]
        )

        # downsampling
        self.conv_in = nn.Conv2d(in_ch, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    SIGEResnetBlock(
                        args=args,
                        config=config,
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        support_sparse=curr_res >= sparse_resolution_threshold,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        SIGEAttnBlock(
                            args=args, config=config, in_channels=block_in, num_heads=num_heads, head_dim=head_dim
                        )
                    )
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = SIGEResnetBlock(
                    args=args,
                    config=config,
                    in_channels=block_in,
                    out_channels=block_in,
                    temb_channels=self.temb_ch,
                    resample="down",
                    support_sparse=curr_res >= sparse_resolution_threshold,
                )
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = SIGEResnetBlock(
            args=args,
            config=config,
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            support_sparse=curr_res >= sparse_resolution_threshold,
        )
        self.mid.attn_1 = SIGEAttnBlock(
            args=args, config=config, in_channels=block_in, num_heads=num_heads, head_dim=head_dim
        )
        self.mid.block_2 = SIGEResnetBlock(
            args=args,
            config=config,
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            support_sparse=curr_res >= sparse_resolution_threshold,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    SIGEResnetBlock(
                        args=args,
                        config=config,
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        support_sparse=curr_res >= sparse_resolution_threshold,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        SIGEAttnBlock(
                            args=args, config=config, in_channels=block_in, num_heads=num_heads, head_dim=head_dim
                        )
                    )
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = SIGEResnetBlock(
                    args=args,
                    config=config,
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    resample="up",
                    support_sparse=curr_res >= sparse_resolution_threshold,
                )
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, logsnr):
        num_resolutions = self.num_resolutions
        ch = self.ch
        num_res_blocks = self.num_res_blocks

        if self.mode == "full":
            logsnr_input_type = self.logsnr_input_type
            if logsnr_input_type == "linear":
                logsnr_input = (logsnr - self.logsnr_scale_range[0]) / (
                    self.logsnr_scale_range[1] - self.logsnr_scale_range[0]
                )
            elif logsnr_input_type == "sigmoid":
                logsnr_input = torch.sigmoid(logsnr)
            elif logsnr_input_type == "inv_cos":
                logsnr_input = torch.arctan(torch.exp(-0.5 * torch.clip(logsnr, -20.0, 20.0))) / (0.5 * math.pi)
            else:
                raise NotImplementedError(self.logsnr_input_type)

            temb = get_timestep_embedding(logsnr_input * 1000, ch)
            temb = self.temb.dense[0](temb)
            temb = swish(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        hs = [self.conv_in(x)]
        for i_level in range(num_resolutions):
            for i_block in range(num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1], temb))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h, temb)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h
