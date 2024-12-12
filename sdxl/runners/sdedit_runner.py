import argparse
import numpy as np

import torch
from einops import repeat

from sige.utils import compute_difference_mask, dilate_mask, downsample_mask
from utils import load_img
from .base_runner import BaseRunner
from sgm.inference.api import SamplingParams, Sampler


class SDEditRunner(BaseRunner):
    @staticmethod
    def modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super(SDEditRunner, SDEditRunner).modify_commandline_options(parser)
        parser.add_argument("--strength", type=float, default=0.8)
        parser.add_argument("--edited_img", type=str, required=True)
        parser.add_argument("--H", type=int, default=512)
        parser.add_argument("--W", type=int, default=512)
        return parser

    def __init__(self, args):
        super().__init__(args)

    def generate(self):
        # Setup
        args = self.args
        device = self.device
        params = SamplingParams()
        params.img2img_strength = args.strength
        is_sige_model = (
            'sige' in args.run_type
        )
        if 'turbo' in args.run_type:
            params.sampler = Sampler.TURBO_SAMPLER
            params.steps = 4

        # Load images and compute difference if necessary
        edited_img = load_img(args.edited_img).to(device)
        edited_img = repeat(edited_img, "1 ... -> b ...", b=1)
        if args.init_img is not None:
            init_img = load_img(args.init_img).to(device)
            init_img = repeat(init_img, "1 ... -> b ...", b=1)
            difference_mask = compute_difference_mask(init_img, edited_img)
            print("Edit Ratio: %.2f%%" % (difference_mask.sum() / difference_mask.numel() * 100))
            difference_mask = dilate_mask(difference_mask, 5)
            masks = downsample_mask(difference_mask, min_res=(4, 4), dilation=1)
        else:
            init_img = None
            masks = None
            difference_mask=None

        # Run diffuser
        samples = self.model.sdedit(
          params = params,
          edited_image=edited_img,
          prompt = args.prompt,
          init_image=init_img,
          masks=masks,
          is_sige_model=is_sige_model,
          difference_mask=difference_mask,
          negative_prompt=args.negative_prompt,
          return_latents=args.refined,
        )

        if args.refined:
            _, samples_z = samples
            del self.model
            torch.cuda.empty_cache()

            samples = self.refiner.sdedit(
                params = params,
                edited_image=samples_z,
                prompt = args.prompt,
                init_image=samples_z,
                masks=masks,
                is_sige_model=is_sige_model,
                difference_mask=difference_mask,
                negative_prompt=args.negative_prompt,
                skip_encode=True,
            )
        self.save_samples(samples)
