import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sige.nn import Gather, Scatter, ScatterGather, ScatterWithBlockResidual, SIGEConv2d, SIGEModel, SIGEModule
from .model import make_attn, my_group_norm, nonlinearity, Normalize


class SIGEResnetBlock(SIGEModule):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        main_block_size=6,
        shortcut_block_size=4,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        if main_block_size is None:
            assert shortcut_block_size is None

        main_support_sparse = main_block_size is not None
        MainConv2d = SIGEConv2d if main_support_sparse else nn.Conv2d

        self.norm1 = Normalize(in_channels)
        self.conv1 = MainConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = MainConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if main_support_sparse:
            self.main_gather = Gather(self.conv1, main_block_size, activation_name="swish")
            self.scatter_gather = ScatterGather(self.main_gather, activation_name="swish")

        if self.in_channels != self.out_channels:
            shortcut_support_sparse = shortcut_block_size is not None
            ShortcutConv2d = SIGEConv2d if shortcut_block_size else nn.Conv2d
            assert not self.use_conv_shortcut
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

        self.scale1, self.shift1 = None, None
        self.scale2, self.shift2 = None, None

    def forward(self, x, temb):
        if self.mode == "full":
            return self.full_forward(x, temb)
        elif self.mode in ("sparse", "profile"):
            return self.sparse_forward(x)
        else:
            raise NotImplementedError("Unknown mode [%s]!!!" % self.mode)

    def full_forward(self, x, temb):
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
        self.scale1, self.shift1 = scale, shift
        h = nonlinearity(h)
        h = self.conv1(h)
        if main_support_sparse:
            h = self.scatter_gather(h)
        if temb is None:
            h, scale, shift = my_group_norm(h, self.norm2)
        else:
            temb = self.temb_proj(nonlinearity(temb))
            temb = temb.view(*temb.shape, 1, 1)
            h = h + temb
            h, scale, shift = my_group_norm(h, self.norm2)
            shift = temb * scale + shift
        self.scale2, self.shift2 = scale, shift
        h = nonlinearity(h)
        h = self.conv2(h)

        if main_support_sparse:
            h = self.scatter(h, x)
        else:
            h = h + x

        return h

    def sparse_forward(self, x):
        main_support_sparse = self.main_support_sparse
        shortcut_support_sparse = self.shortcut_support_sparse

        h = x
        if self.in_channels != self.out_channels:
            if shortcut_support_sparse:
                x = self.shortcut_gather(x)
            x = self.nin_shortcut(x)
        if main_support_sparse:
            h = self.main_gather(h, self.scale1, self.shift1)
        else:
            h = h * self.scale1 + self.shift1
            h = nonlinearity(h)
        h = self.conv1(h)

        if main_support_sparse:
            h = self.scatter_gather(h, self.scale2, self.shift2)
        else:
            h = h * self.scale2 + self.shift2
            h = nonlinearity(h)

        h = self.conv2(h)

        if main_support_sparse:
            h = self.scatter(h, x)
        else:
            h = h + x
        return h


class SIGEDownsample(SIGEModule):
    def __init__(self, in_channels, with_conv, block_size: int = 6):
        super(SIGEDownsample, self).__init__()
        assert with_conv
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
        self.gather = Gather(self.conv, block_size=block_size)
        self.scatter = Scatter(self.gather)

    def forward(self, x):
        x = self.gather(x)
        if self.mode == "full":
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        x = self.scatter(x)
        return x


class SIGEUpsample(SIGEModule):
    def __init__(self, in_channels, with_conv, block_size=6):
        super(SIGEUpsample, self).__init__()
        assert with_conv
        self.conv = SIGEConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.gather = Gather(self.conv, block_size=block_size)
        self.scatter = Scatter(self.gather)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.gather(x)
        x = self.conv(x)
        x = self.scatter(x)
        return x


class SIGEEncoder(SIGEModel):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type="sige",
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    SIGEResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = SIGEDownsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = SIGEResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = SIGEResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class SIGEDecoder(SIGEModel):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type="sige",
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = SIGEResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = SIGEResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    SIGEResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = SIGEUpsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
