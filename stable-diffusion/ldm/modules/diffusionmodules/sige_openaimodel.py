from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from sige.nn import Gather, Scatter, ScatterGather, ScatterWithBlockResidual, SIGEConv2d, SIGEModel, SIGEModule
from .openaimodel import ResBlock, TimestepBlock, TimestepEmbedSequential, UNetModel
from .sige_model import my_group_norm
from .util import normalization, zero_module
from ..sige_attention import SIGESpatialTransformer


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
            emb_scale, emb_shift = torch.chunk(emb_out, 2, dim=1)
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
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
    ):
        super(UNetModel, self).__init__()
        SIGEModel.__init__(self, call_super=False)
        assert use_spatial_transformer
        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    SIGEResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        main_block_size=6,
                        shortcut_block_size=4,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        SIGESpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(SIGEDownsample(ch, conv_resample, dims=dims, out_channels=out_ch))
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
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SIGESpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                block_size=None,
                use_checkpoint=use_checkpoint,
            ),
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
            for i in range(num_res_blocks + 1):
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
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        SIGESpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(SIGEUpsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                nn.Conv2d(model_channels, n_embed, 1),
                # nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
            )
