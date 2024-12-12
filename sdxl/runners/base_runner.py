import argparse
import os

import numpy as np
import torch
from einops import rearrange
from imwatermark import WatermarkEncoder
from omegaconf import OmegaConf
from PIL import Image

# from sgm.models.diffusion import DiffusionEngine
from utils import check_safety, put_watermark
from sgm.inference.api import SamplingPipeline, ModelArchitecture

import copy

class NamespaceCopy:
    def __init__(self, namespace):
        self.args = copy.deepcopy(namespace.__dict__)
    
    def __getattr__(self, x):
        return self.args[x]


class BaseRunner:
    @staticmethod
    def modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--mode",
            type=str,
            default="generate",
            choices=["generate", "profile_unet", "profile_encoder", "profile_decoder"],
        )
        parser.add_argument("--prompt", type=str, required=True)
        parser.add_argument("--negative_prompt", type=str, default="")
        parser.add_argument("--output_path", type=str, default=None)
        parser.add_argument("--ddim_steps", type=int, default=50)
        parser.add_argument("--ddim_eta", type=float, default=0.0)
        parser.add_argument("--config_path", type=str, default="configs")
        parser.add_argument("--weight_path", type=str, default="pretrained")
        parser.add_argument("--run_type", type=str, default="original", choices=['original', 'sige', 'turbo', 'sige-turbo'])
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--scale", type=float, default=7.5)
        parser.add_argument("--seed", type=int, default=2)
        parser.add_argument("--init_img", type=str, default=None)
        parser.add_argument("--C", type=int, default=4)
        parser.add_argument("--f", type=int, default=8)
        parser.add_argument("--refined", type=bool, default=False)
        return parser

    def __init__(self, args):
        self.args = args

        if args.device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        elif args.device == "cpu":
            device = torch.device("cpu")
        elif args.device == "cuda":
            device = torch.device("cuda")
        else:
            raise NotImplementedError("Unknown device [%s]!!!" % args.device)
        run_type = args.run_type

        wm = "SDXL"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark("bytes", wm.encode("utf-8"))

        self.device = device
        self.wm_encoder = wm_encoder

        # Get architecture
        if run_type == 'original':
            architecture = ModelArchitecture.SDXL_V1_BASE
        elif run_type == 'sige':
            architecture = ModelArchitecture.SDXL_SIGE_V1_BASE
        elif run_type == 'turbo':
            architecture = ModelArchitecture.SDXL_TURBO
        elif run_type == 'sige-turbo':
            architecture = ModelArchitecture.SDXL_SIGE_TURBO
        else:
            raise NotImplementedError("Unknown architecture [%s]!!!" % run_type)
        
        if args.mode != 'generate' and args.refined:
            refined_args = NamespaceCopy(args)
            args.mode = 'generate'
        else:
            refined_args = args

        self.model = SamplingPipeline(
            model_id=architecture,
            model_path=args.weight_path,
            config_path=args.config_path,
            device=device,
            mask_path=args.mask_path if 'mask_path' in args else None,
            args=args,
            use_fp16=False
        )

        if args.refined:
            assert run_type in ['original', 'sige'], "Don't support refiner for Turbo"
            self.refiner = SamplingPipeline(
                model_id=ModelArchitecture.SDXL_SIGE_V1_REFINER if run_type == 'sige' else ModelArchitecture.SDXL_V1_REFINER,
                model_path=args.weight_path,
                config_path=args.config_path,
                device=device,
                mask_path=args.mask_path if 'mask_path' in args else None,
                args=refined_args,
                use_fp16=False
            )
        else:
            self.refiner = None

    def generate(self):
        raise NotImplementedError

    def run(self):
        self.generate()

    def save_samples(self, samples):
        args = self.args
        samples = samples.cpu().permute(0, 2, 3, 1).numpy()
        checked_image, _ = check_safety(samples)
        checked_image_torch = torch.from_numpy(checked_image)
        checked_image_torch = checked_image_torch.permute(0, 3, 1, 2)
        for i, sample in enumerate(checked_image_torch):
            sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
            img = Image.fromarray(sample.astype(np.uint8))
            img = put_watermark(img, self.wm_encoder)
            if args.output_path is not None:
                os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
                img.save(args.output_path)
