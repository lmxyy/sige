import logging
import math
from abc import abstractmethod
from typing import Iterable, List, Optional, Tuple, Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from sige.nn import Gather, Scatter, ScatterGather, ScatterWithBlockResidual, SIGEConv2d, SIGEModel, SIGEModule
from .openaimodel import ResBlock, TimestepBlock, TimestepEmbedSequential, UNetModel, Timestep
from ..sige_attention import SIGESpatialTransformer
from ...modules.diffusionmodules.util import (avg_pool_nd, conv_nd, linear,
                                              normalization,
                                              timestep_embedding, zero_module)
from .sige_model import my_group_norm
from ...modules.video_attention import SpatialVideoTransformer
from ...util import exists

logpy = logging.getLogger(__name__)


class SIGEDownsample(SIGEModule):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, block_size=6):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        assert dims == 2
        stride = 2 if dims != 3 else (1, 2, 2)
        assert use_conv
        self.op = SIGEConv2d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        self.gather = Gather(self.op, block_size=block_size)
        self.scatter = Scatter(self.gather)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = self.gather(x)
        x = self.op(x)
        x = self.scatter(x)
        return x


class SIGEUpsample(SIGEModule):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, block_size=6):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        assert dims == 2
        assert use_conv
        self.conv = SIGEConv2d(self.channels, self.out_channels, 3, padding=padding)
        self.gather = Gather(self.conv, block_size=block_size)
        self.scatter = Scatter(self.gather)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.gather(x)
        x = self.conv(x)
        x = self.scatter(x)
        return x


class SIGEResBlock(TimestepBlock, SIGEModule):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        main_block_size: Optional[int] = 6,
        shortcut_block_size: Optional[int] = 4,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        assert dims == 2

        main_support_sparse = main_block_size is not None
        MainConv2d = SIGEConv2d if main_support_sparse else nn.Conv2d

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            MainConv2d(channels, self.out_channels, 3, padding=1),
        )

        assert not up and not down
        self.updown = up or down

        self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(MainConv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if main_support_sparse:
            self.main_gather = Gather(self.in_layers[2], main_block_size, activation_name="swish")
            self.scatter_gather = ScatterGather(self.main_gather, activation_name="swish")

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
            shortcut_support_sparse = False
            if main_support_sparse:
                self.scatter = Scatter(self.main_gather)
        elif use_conv:
            assert False
        else:
            shortcut_support_sparse = shortcut_block_size is not None
            ShortcutConv2d = SIGEConv2d if shortcut_block_size else nn.Conv2d
            self.skip_connection = ShortcutConv2d(channels, self.out_channels, 1)
            if shortcut_support_sparse:
                self.shortcut_gather = Gather(self.skip_connection, shortcut_block_size)
                self.scatter = ScatterWithBlockResidual(self.main_gather, self.shortcut_gather)
            elif main_support_sparse:
                self.scatter = Scatter(self.main_gather)
        self.main_support_sparse = main_support_sparse
        self.shortcut_support_sparse = shortcut_support_sparse

        self.scale1, self.shift1 = None, None
        self.scale2, self.shift2 = None, None

    def forward(self, x, emb):
        if self.mode == "full":
            return self.full_forward(x, emb)
        elif self.mode in ("sparse", "profile"):
            return self.sparse_forward(x)
        else:
            raise NotImplementedError("Unknown mode [%s]!!!" % self.mode)

    def full_forward(self, x, emb):
        main_support_sparse = self.main_support_sparse
        shortcut_support_sparse = self.shortcut_support_sparse

        h = x
        if self.channels != self.out_channels:
            if shortcut_support_sparse:
                x = self.shortcut_gather(x)
            x = self.skip_connection(x)

        if main_support_sparse:
            h = self.main_gather(h)
        h, scale, shift = my_group_norm(h, self.in_layers[0])
        self.scale1, self.shift1 = scale, shift
        h = self.in_layers[1](h)
        h = self.in_layers[2](h)
        if main_support_sparse:
            h = self.scatter_gather(h)
        emb_out = self.emb_layers(emb).type(h.dtype)
        emb_out = emb_out.view(*emb_out.shape, 1, 1)
        if self.use_scale_shift_norm:
            h, norm_scale, norm_shift = my_group_norm(h, self.out_layers[0])
            emb_scale, emb_shift = th.chunk(emb_out, 2, dim=1)
            h = h * (1 + emb_scale) + emb_shift
            scale = norm_scale * (1 + emb_scale)
            shift = norm_shift * (1 + emb_scale) + emb_shift
        else:
            h = h + emb_out
            h, norm_scale, norm_shift = my_group_norm(h, self.out_layers[0])
            scale = norm_scale
            shift = norm_scale * emb_out + norm_shift
        self.scale2, self.shift2 = scale, shift
        h = self.out_layers[1](h)
        h = self.out_layers[2](h)
        h = self.out_layers[3](h)
        if main_support_sparse:
            h = self.scatter(h, x)
        else:
            h = h + x
        return h

    def sparse_forward(self, x):
        main_support_sparse = self.main_support_sparse
        shortcut_support_sparse = self.shortcut_support_sparse

        h = x
        if self.channels != self.out_channels:
            if shortcut_support_sparse:
                x = self.shortcut_gather(x)
            x = self.skip_connection(x)
        if main_support_sparse:
            h = self.main_gather(h, self.scale1, self.shift1)
        else:
            h = h * self.scale1 + self.shift1
            h = self.in_layers[1](h)
        h = self.in_layers[2](h)

        if main_support_sparse:
            h = self.scatter_gather(h, self.scale2, self.shift2)
        else:
            h = h * self.scale2 + self.shift2
            h = self.out_layers[1](h)
        h = self.out_layers[2](h)
        h = self.out_layers[3](h)
        if main_support_sparse:
            h = self.scatter(h, x)
        else:
            h = h + x
        return h


class SIGEUNetModel(SIGEModel, UNetModel):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: int,
        dropout: float = 0.0,
        channel_mult: Union[List, Tuple] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[Union[int, str]] = None,
        use_checkpoint: bool = False,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        transformer_depth: int = 1,
        context_dim: Optional[int] = None,
        disable_self_attentions: Optional[List[bool]] = None,
        num_attention_blocks: Optional[List[int]] = None,
        disable_middle_self_attn: bool = False,
        disable_middle_transformer: bool = False,
        use_linear_in_transformer: bool = False,
        spatial_transformer_attn_type: str = "softmax",
        adm_in_channels: Optional[int] = None,
    ):
        super(UNetModel, self).__init__()
        SIGEModel.__init__(self, call_super=False)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = transformer_depth[-1]

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks

        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            logpy.info(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.dtype = th.float32  # Don't support float16

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                logpy.info("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    SIGEResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if context_dim is not None and exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            SIGESpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth[level],
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                attn_type=spatial_transformer_attn_type,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        SIGEResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else SIGEDownsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                out_channels=ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SIGESpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                attn_type=spatial_transformer_attn_type,
                use_checkpoint=use_checkpoint,
            )
            if not disable_middle_transformer
            else th.nn.Identity(),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    SIGEResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or i < num_attention_blocks[level]
                    ):
                        layers.append(
                            SIGESpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth[level],
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                attn_type=spatial_transformer_attn_type,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        SIGEResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else SIGEUpsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
