import pytorch_lightning as pl
import torch

from ldm.modules.diffusionmodules.model import Decoder, Encoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config


class AutoencoderKL(pl.LightningModule):
    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
    ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
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

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        if self.args.mode == "profile_encoder":
            if not hasattr(self.encoder, "mode") or self.encoder.mode == "sparse":
                if hasattr(self.encoder, "mode"):
                    self.encoder.set_mode("profile")
                from torchprofile import profile_macs

                macs = profile_macs(self.encoder, (x,))
                print("MACs: %.3fG" % (macs / 1e9))

                from tqdm import trange
                import time

                if hasattr(self.encoder, "mode"):
                    self.encoder.set_mode("sparse")
                for _ in trange(100):
                    self.encoder(x)
                    torch.cuda.synchronize()
                start = time.time()
                for _ in trange(100):
                    self.encoder(x)
                    torch.cuda.synchronize()
                print(f"Time per forward pass: {(time.time() - start) * 10} ms\n\n\n")
                exit(0)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        if self.args.mode == "profile_decoder":
            if not hasattr(self.decoder, "mode") or self.decoder.mode == "sparse":
                if hasattr(self.decoder, "mode"):
                    self.decoder.set_mode("profile")
                from torchprofile import profile_macs

                macs = profile_macs(self.decoder, (z,))
                print("MACs: %.3fG" % (macs / 1e9))

                from tqdm import trange
                import time

                if hasattr(self.decoder, "mode"):
                    self.decoder.set_mode("sparse")
                for _ in trange(100):
                    self.decoder(z)
                    torch.cuda.synchronize()
                start = time.time()
                for _ in trange(100):
                    self.decoder(z)
                    torch.cuda.synchronize()
                print(f"Time per forward pass: {(time.time() - start) * 10} ms\n\n\n")
                exit(0)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior
