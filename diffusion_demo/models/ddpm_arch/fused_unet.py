"""
Fuse the parallel temb FC layers and the qkv convolutions in the attention module.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchprofile import profile_macs

from .unet import AttnBlock, Downsample, ResnetBlock, UNet, Upsample
from ..common import Normalize, get_timestep_embedding, swish


class FusedResnetBlock(nn.Module):
    def __init__(self, args, config, in_channels, out_channels=None, conv_shortcut=False, dropout=0.1):
        super(FusedResnetBlock, self).__init__()
        self.args = args
        self.config = config
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.dropout_rate = dropout

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)

        h = swish(h)
        h = self.conv1(h)

        h = h + temb[:, :, None, None]

        h = self.norm2(h)

        h = swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

    def profile_forward(self, x, temb):
        return profile_macs(self, (x, temb))

    @classmethod
    def from_ResnetBlock(cls, resnet_block: ResnetBlock):
        ret = cls(
            resnet_block.args,
            resnet_block.config,
            resnet_block.in_channels,
            resnet_block.out_channels,
            resnet_block.use_conv_shortcut,
            resnet_block.dropout_rate,
        )

        ret.norm1 = resnet_block.norm1
        ret.conv1 = resnet_block.conv1
        ret.norm2 = resnet_block.norm2
        ret.conv2 = resnet_block.conv2
        if ret.in_channels != ret.out_channels:
            if ret.use_conv_shortcut:
                ret.conv_shortcut = resnet_block.conv_shortcut
            else:
                ret.nin_shortcut = resnet_block.nin_shortcut
        return ret


class FusedAttnBlock(nn.Module):
    def __init__(self, args, config, in_channels):
        super(FusedAttnBlock, self).__init__()
        self.args = args
        self.config = config
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.qkv = nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)

        qkv = self.qkv(h_)
        q = qkv[:, : self.in_channels]
        k = qkv[:, self.in_channels : 2 * self.in_channels]
        v = qkv[:, 2 * self.in_channels :]

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

        h_ = self.proj_out(h_)

        return x + h_

    @classmethod
    def from_AttnBlock(cls, attn_block: AttnBlock):
        ret = cls(attn_block.args, attn_block.config, attn_block.in_channels)
        ret.norm = attn_block.norm
        ret.proj_out = attn_block.proj_out
        qw = attn_block.q.weight.data
        kw = attn_block.k.weight.data
        vw = attn_block.v.weight.data
        qb = attn_block.q.bias.data
        kb = attn_block.k.bias.data
        vb = attn_block.v.bias.data

        ret.qkv.weight.data = torch.cat([qw, kw, vw], dim=0)  # OIHW
        ret.qkv.bias.data = torch.cat([qb, kb, vb], dim=0)
        return ret


class FusedUNet(nn.Module):
    def __init__(self, args, config):
        super(FusedUNet, self).__init__()
        self.args = args
        self.config = config

        ch, ch_mult = config.model.ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_ch, out_ch = config.model.in_ch, config.model.out_ch
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv

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
                    FusedResnetBlock(
                        args=args, config=config, in_channels=block_in, out_channels=block_out, dropout=dropout
                    )
                )
                temb_proj_dim += block_out
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(FusedAttnBlock(args=args, config=config, in_channels=block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(args=args, config=config, in_channels=block_in, with_conv=resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = FusedResnetBlock(
            args=args, config=config, in_channels=block_in, out_channels=block_in, dropout=dropout
        )
        temb_proj_dim += block_in
        self.mid.attn_1 = FusedAttnBlock(args=args, config=config, in_channels=block_in)
        self.mid.block_2 = FusedResnetBlock(
            args=args, config=config, in_channels=block_in, out_channels=block_in, dropout=dropout
        )
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
                    FusedResnetBlock(
                        args=args,
                        config=config,
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                temb_proj_dim += block_out
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(FusedAttnBlock(args=args, config=config, in_channels=block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(args=args, config=config, in_channels=block_in, with_conv=resamp_with_conv)
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
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = swish(temb)
        temb = self.temb.dense[1](temb)
        temb = swish(temb)
        temb = self.temb.dense[2](temb)

        offset = 0

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                block_out = self.down[i_level].block[i_block].out_channels
                t = temb[:, offset : offset + block_out]
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
        h = self.mid.block_1(h, temb[:, offset : offset + block_out])
        offset += block_out
        h = self.mid.attn_1(h)
        block_out = self.mid.block_2.out_channels
        h = self.mid.block_2(h, temb[:, offset : offset + block_out])
        offset += block_out

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                block_out = self.up[i_level].block[i_block].out_channels
                t = temb[:, offset : offset + block_out]
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

    @classmethod
    def from_unet(cls, unet: UNet):
        ret = cls(unet.args, unet.config)
        ret.temb.dense[0] = unet.temb.dense[0]
        ret.temb.dense[1] = unet.temb.dense[1]
        ret.conv_in = unet.conv_in

        def transfer_linear(fc1: nn.Linear, fc2: nn.Linear, offset):
            c = fc2.out_features
            fc1.weight.data[offset : offset + c] = fc2.weight.data  # OI
            fc1.bias.data[offset : offset + c] = fc2.bias.data
            return offset + c

        offset = 0

        for i_level in range(ret.num_resolutions):
            for i_block in range(ret.num_res_blocks):
                resnet_block = unet.down[i_level].block[i_block]
                ret.down[i_level].block[i_block] = FusedResnetBlock.from_ResnetBlock(resnet_block)
                offset = transfer_linear(ret.temb.dense[2], resnet_block.temb_proj, offset)
                if len(ret.down[i_level].attn) > 0:
                    ret.down[i_level].attn[i_block] = FusedAttnBlock.from_AttnBlock(unet.down[i_level].attn[i_block])
            if i_level != ret.num_resolutions - 1:
                ret.down[i_level].downsample = unet.down[i_level].downsample

        # middle
        resnet_block = unet.mid.block_1
        ret.mid.block_1 = FusedResnetBlock.from_ResnetBlock(resnet_block)
        offset = transfer_linear(ret.temb.dense[2], resnet_block.temb_proj, offset)
        ret.mid.attn_1 = FusedAttnBlock.from_AttnBlock(unet.mid.attn_1)
        resnet_block = unet.mid.block_2
        ret.mid.block_2 = FusedResnetBlock.from_ResnetBlock(resnet_block)
        offset = transfer_linear(ret.temb.dense[2], resnet_block.temb_proj, offset)

        # upsampling
        for i_level in reversed(range(ret.num_resolutions)):
            for i_block in range(ret.num_res_blocks + 1):
                resnet_block = unet.up[i_level].block[i_block]
                ret.up[i_level].block[i_block] = FusedResnetBlock.from_ResnetBlock(resnet_block)
                offset = transfer_linear(ret.temb.dense[2], resnet_block.temb_proj, offset)
                if len(ret.up[i_level].attn) > 0:
                    ret.up[i_level].attn[i_block] = FusedAttnBlock.from_AttnBlock(unet.up[i_level].attn[i_block])
            if i_level != 0:
                ret.up[i_level].upsample = unet.up[i_level].upsample

        # end
        ret.norm_out = unet.norm_out
        ret.conv_out = unet.conv_out

        return ret
