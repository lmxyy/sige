import argparse

import numpy as np
import torch
from einops import repeat

from ldm.models.sige_autoencoder import SIGEAutoencoderKL
from sige.nn import SIGEModel
from sige.utils import downsample_mask
from utils import load_img
from .base_runner import BaseRunner


class InpaintingRunner(BaseRunner):
    @staticmethod
    def modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super(InpaintingRunner, InpaintingRunner).modify_commandline_options(parser)
        parser.add_argument("--H", type=int, default=512)
        parser.add_argument("--W", type=int, default=512)
        parser.add_argument("--mask_path", type=str, required=True)
        return parser

    def __init__(self, args):
        super().__init__(args)
        assert args.init_img is not None, "Must provide an initial image for inpainting"

    def generate(self):
        args = self.args
        model = self.model
        sampler = self.sampler
        device = self.device

        prompts = [args.prompt]
        if args.scale != 1.0:
            uc = model.get_learned_conditioning([""])
        c = model.get_learned_conditioning(prompts)

        init_img = load_img(args.init_img).to(device)
        init_img = repeat(init_img, "1 ... -> b ...", b=1)

        if isinstance(model.first_stage_model, SIGEAutoencoderKL):
            assert isinstance(model.first_stage_model.encoder, SIGEModel)
            model.first_stage_model.encoder.set_mode("full")
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_img))

        del model.first_stage_model.encoder
        model.first_stage_model.encoder = None
        torch.cuda.empty_cache()

        shape = (args.C, args.H // args.f, args.W // args.f)

        mask = np.load(args.mask_path)
        mask = torch.from_numpy(mask).to(device)
        masks = downsample_mask(mask, min_res=8, dilation=1)

        samples, _ = sampler.sample(
            S=args.ddim_steps,
            conditioning=c,
            batch_size=1,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=args.scale,
            unconditional_conditioning=uc,
            eta=args.ddim_eta,
            x_T=None,
            mask=1 - masks[tuple(shape[1:])][None, None].float(),
            x0=init_latent,
            conv_masks=masks,
        )
        if isinstance(model.first_stage_model, SIGEAutoencoderKL):
            assert isinstance(model.first_stage_model.decoder, SIGEModel)
            model.first_stage_model.decoder.set_mode("full")
            model.decode_first_stage(init_latent)
            model.first_stage_model.decoder.set_masks(masks)
            model.first_stage_model.decoder.set_mode("sparse")
        samples = model.decode_first_stage(samples)
        self.save_samples(samples)
