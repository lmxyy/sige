import argparse
import os

import numpy as np
import torch
from einops import rearrange
from imwatermark import WatermarkEncoder
from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from utils import check_safety, load_model_from_config, put_watermark


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
        parser.add_argument("--output_path", type=str, default=None)
        parser.add_argument("--ddim_steps", type=int, default=50)
        parser.add_argument("--ddim_eta", type=float, default=0.0)
        parser.add_argument("--config_path", type=str, default="configs/original.yaml")
        parser.add_argument("--weight_path", type=str, default="pretrained/sd-v1-4.ckpt")
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--scale", type=float, default=7.5)
        parser.add_argument("--seed", type=int, default=2)
        parser.add_argument("--init_img", type=str, default=None)
        parser.add_argument("--C", type=int, default=4)
        parser.add_argument("--f", type=int, default=8)
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

        config = OmegaConf.load(args.config_path)
        model = load_model_from_config(config, args.weight_path)
        assert isinstance(model, LatentDiffusion)
        model = model.to(device)
        model.eval()

        model.model.args = args
        model.first_stage_model.args = args

        sampler = DDIMSampler(args, model)
        sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=args.ddim_eta, verbose=False)

        wm = "StableDiffusionV1"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark("bytes", wm.encode("utf-8"))

        self.device = device
        self.config = config
        self.model = model
        self.sampler = sampler
        self.wm_encoder = wm_encoder

    def generate(self):
        raise NotImplementedError

    def run(self):
        model = self.model

        with torch.no_grad():
            with model.ema_scope():
                self.generate()

    def save_samples(self, samples):
        args = self.args
        samples = torch.clamp((samples + 1) / 2, min=0, max=1)
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
