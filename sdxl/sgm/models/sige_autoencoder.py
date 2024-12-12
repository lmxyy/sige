import torch

from sgm.modules.diffusionmodules.sige_model import SIGEDecoder, SIGEEncoder
from sgm.util import instantiate_from_config
from .autoencoder import AutoencoderKL


class SIGEAutoencoderKL(AutoencoderKL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ddconfig = kwargs.pop('ddconfig')
        lossconfig = kwargs.pop('lossconfig')
        self.image_key = kwargs.pop('image_key', 'image')
        self.encoder = SIGEEncoder(**ddconfig)
        self.decoder = SIGEDecoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        embed_dim = kwargs.pop('embed_dim')
        colorize_nlabels = kwargs.pop("colorize_nlabels", None)
        monitor = kwargs.pop("monitor", None)
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])

        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
