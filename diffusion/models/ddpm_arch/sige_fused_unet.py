import torch
from torch import nn
from torch.nn import functional as F

from sige.nn import Gather, Scatter, ScatterGather, ScatterWithBlockResidual, SIGEConv2d, SIGEModel, SIGEModule
from .unet import Downsample
from ..common import get_timestep_embedding, my_group_norm, Normalize, swish


class SIGEFusedResnetBlock(SIGEModule):
    def __init__(self, args, config, in_channels, out_channels=None, support_sparse=False):
        super(SIGEFusedResnetBlock, self).__init__()
        self.args = args
        self.config = config
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.support_sparse = support_sparse

        main_block_size = config.model.sige_block_size.normal
        main_support_sparse = support_sparse and main_block_size is not None

        MainConv2d = SIGEConv2d if main_support_sparse else nn.Conv2d
        self.norm1 = Normalize(in_channels)
        self.conv1 = MainConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.conv2 = MainConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if main_support_sparse:
            self.main_gather = Gather(self.conv1, main_block_size, activation_name="swish")
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
            shortcut_support_sparse = False
            if main_support_sparse:
                self.scatter = Scatter(self.main_gather)

        self.main_support_sparse = main_support_sparse
        self.shortcut_support_sparse = shortcut_support_sparse

        self.scale1s, self.shift1s = {}, {}
        self.scale2s, self.shift2s = {}, {}

    def clear_cache(self):
        self.scale1s, self.shift1s = {}, {}
        self.scale2s, self.shift2s = {}, {}

    def forward(self, x, temb):
        if self.mode == "full":
            return self.full_forward(x, temb)
        elif self.mode in ("sparse", "profile"):
            return self.sparse_forward(x)
        else:
            raise NotImplementedError("Unknown mode [%s]!!!" % self.mode)

    def full_forward(self, x, temb):
        cache_id = self.cache_id
        main_support_sparse = self.main_support_sparse
        shortcut_support_sparse = self.shortcut_support_sparse

        h = x
        if self.in_channels != self.out_channels:
            if shortcut_support_sparse:
                x = self.shortcut_gather(x)
            x = self.nin_shortcut(x)

        if main_support_sparse:
            h = self.main_gather(h)  # record the input resolution
        h, scale, shift = my_group_norm(h, self.norm1)
        self.scale1s[cache_id], self.shift1s[cache_id] = scale, shift
        h = swish(h)
        h = self.conv1(h)
        if main_support_sparse:
            h = self.scatter_gather(h)
        h = h + temb.view(*temb.shape, 1, 1)
        temb = temb.view(-1)
        h, scale, shift = my_group_norm(h, self.norm2)
        shift = temb * scale + shift
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
        if self.in_channels != self.out_channels:
            if shortcut_support_sparse:
                x = self.shortcut_gather(x)
            x = self.nin_shortcut(x)
        if main_support_sparse:
            h = self.main_gather(h, self.scale1s[cache_id].view(1, -1, 1, 1), self.shift1s[cache_id].view(1, -1, 1, 1))
        else:
            h = h * self.scale1s[cache_id].view(1, -1, 1, 1) + self.shift1s[cache_id].view(1, -1, 1, 1)
            h = swish(h)
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


class SIGEFusedAttnBlock(SIGEModule):
    def __init__(self, args, config, in_channels, support_sparse=False):
        super(SIGEFusedAttnBlock, self).__init__()
        self.args = args
        self.config = config
        self.in_channels = in_channels

        block_size = config.model.sige_block_size.instance
        support_sparse = support_sparse and block_size is not None
        self.support_sparse = support_sparse

        self.norm = Normalize(in_channels)

        Conv2d = SIGEConv2d if support_sparse else nn.Conv2d
        self.qkv = Conv2d(in_channels, 3 * in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

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
            self.scales, self.shifts = scale, shift
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
        q, k, v = torch.split(qkv, self.in_channels, dim=1)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        if self.support_sparse:
            h_ = self.gather2(h_)
        h_ = self.proj_out(h_)

        if self.support_sparse:
            h_ = self.scatter2(h_, x)
        else:
            h_ = h_ + x
        return h_


class SIGEUpsample(SIGEModule):
    def __init__(self, args, config, in_channels, with_conv):
        super(SIGEUpsample, self).__init__()
        self.args = args
        self.config = config
        assert with_conv
        self.conv = SIGEConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.gather = Gather(self.conv, block_size=config.model.sige_block_size.normal)
        self.scatter = Scatter(self.gather)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.gather(x)
        x = self.conv(x)
        x = self.scatter(x)
        return x


class SIGEDownsample(SIGEModule):
    def __init__(self, args, config, in_channels, with_conv):
        super(SIGEDownsample, self).__init__()
        self.args = args
        self.config = config
        assert with_conv
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
        self.gather = Gather(self.conv, block_size=config.model.sige_block_size.normal)
        self.scatter = Scatter(self.gather)

    def forward(self, x):
        x = self.gather(x)
        if self.mode in ["full"]:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        x = self.scatter(x)
        return x


class SIGEFusedUNet(SIGEModel):
    def __init__(self, args, config):
        super(SIGEFusedUNet, self).__init__()
        self.args = args
        self.config = config

        ch, ch_mult = config.model.ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        in_ch, out_ch = config.model.in_ch, config.model.out_ch
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        sparse_resolution_threshold = config.model.sparse_resolution_threshold

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList(
            [
                nn.Linear(self.ch, self.temb_ch),
                nn.Linear(self.temb_ch, self.temb_ch),
            ]
        )
        temb_proj_dim = 0

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
                    SIGEFusedResnetBlock(
                        args=args,
                        config=config,
                        in_channels=block_in,
                        out_channels=block_out,
                        support_sparse=curr_res >= sparse_resolution_threshold,
                    )
                )
                temb_proj_dim += block_out
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        SIGEFusedAttnBlock(
                            args=args,
                            config=config,
                            in_channels=block_in,
                            support_sparse=curr_res >= sparse_resolution_threshold,
                        )
                    )
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                DownsampleClass = SIGEDownsample if curr_res >= sparse_resolution_threshold else Downsample
                down.downsample = DownsampleClass(
                    args=args, config=config, in_channels=block_in, with_conv=resamp_with_conv
                )
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = SIGEFusedResnetBlock(args=args, config=config, in_channels=block_in, out_channels=block_in)
        temb_proj_dim += block_in
        self.mid.attn_1 = SIGEFusedAttnBlock(args=args, config=config, in_channels=block_in)
        self.mid.block_2 = SIGEFusedResnetBlock(args=args, config=config, in_channels=block_in, out_channels=block_in)
        temb_proj_dim += block_in

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
                    SIGEFusedResnetBlock(
                        args=args,
                        config=config,
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        support_sparse=curr_res >= sparse_resolution_threshold,
                    )
                )
                temb_proj_dim += block_out
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        SIGEFusedAttnBlock(
                            args=args,
                            config=config,
                            in_channels=block_in,
                            support_sparse=curr_res >= sparse_resolution_threshold,
                        )
                    )
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = SIGEUpsample(args=args, config=config, in_channels=block_in, with_conv=resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        self.temb.dense.append(nn.Linear(self.temb_ch, temb_proj_dim))
        self.temb_proj_dim = temb_proj_dim

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        if self.mode == "full":
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = swish(temb)
            temb = self.temb.dense[1](temb)
            temb = swish(temb)
            temb = self.temb.dense[2](temb)
        else:
            temb = None

        offset = 0

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                block_out = self.down[i_level].block[i_block].out_channels
                t = temb[:, offset : offset + block_out] if self.mode == "full" else None
                h = self.down[i_level].block[i_block](hs[-1], t)
                offset += block_out
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        block_out = self.mid.block_1.out_channels
        h = self.mid.block_1(h, temb[:, offset : offset + block_out] if self.mode == "full" else None)
        offset += block_out
        h = self.mid.attn_1(h)
        block_out = self.mid.block_2.out_channels
        h = self.mid.block_2(h, temb[:, offset : offset + block_out] if self.mode == "full" else None)
        offset += block_out

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                block_out = self.up[i_level].block[i_block].out_channels
                t = temb[:, offset : offset + block_out] if self.mode == "full" else None
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), t)
                offset += block_out
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h
