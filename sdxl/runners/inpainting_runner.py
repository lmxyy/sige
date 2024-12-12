import argparse

import numpy as np
import torch
from einops import repeat

from sige.utils import downsample_mask
from utils import load_img
from .base_runner import BaseRunner
from sgm.inference.api import SamplingParams, Sampler


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
        # Setup
        args = self.args
        device = self.device

        params = SamplingParams()
        params.width = args.W
        params.height = args.H
        if 'turbo' in args.run_type:
            params.sampler = Sampler.TURBO_SAMPLER
            params.steps = 4

        # Generate the masks
        shape = (args.C, args.H // args.f, args.W // args.f)
        mask = np.load(args.mask_path)
        mask = torch.from_numpy(mask).to(device)
        masks = downsample_mask(mask, min_res=8, dilation=1)
        mask = 1 - masks[tuple(shape[1:])][None, None].float()
        conv_masks=masks

        # Run diffuser
        samples = self.model.inpaint(
          params = params,
          image = load_img(args.init_img).to(device),
          prompt = args.prompt,
          mask=mask,
          conv_masks=conv_masks,
          negative_prompt=args.negative_prompt,
          return_latents=args.refined,
        )

        if args.refined:
            _, samples_z = samples
            del self.model
            torch.cuda.empty_cache()

            samples = self.refiner.inpaint(
                params=params,
                image=samples_z,
                prompt=args.prompt,
                mask=mask,
                conv_masks=conv_masks,
                negative_prompt=args.negative_prompt,
                skip_encode=True,
            )

        self.save_samples(samples)
