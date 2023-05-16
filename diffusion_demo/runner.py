from typing import Optional

import numpy as np
import torch
from torch import nn

from download_helper import get_ckpt_path
from models.ema import EMAHelper
from samplers.base_sampler import BaseSampler
from sige.nn import SIGEModel
from sige.utils import compute_difference_mask, dilate_mask, downsample_mask
from utils import data_transform, inverse_data_transform, set_seed


class Runner:
    def __init__(self, args, config):
        self.args, self.config = args, config

        self.device = config.device
        self.sampler = self.get_sampler()

        model, ema_helper = self.build_model()
        pretrained_path = get_ckpt_path(config, tool=args.download_tool)
        args.restore_from = pretrained_path
        self.restore_checkpoint(model, ema_helper=ema_helper)
        if ema_helper is not None:
            ema_helper.ema(model)

        model.eval()

        self.model = model
        self.seq = self.get_sampling_sequence(config.sampling.noise_level)
        self.noise = None

    def get_sampler(self) -> BaseSampler:
        args, config = self.args, self.config
        sampler_type = config.sampling.sampler_type
        if sampler_type == "ddim":
            from samplers.ddim_sampler import DDIMSampler as Sampler
        elif sampler_type == "ddpm":
            from samplers.ddpm_sampler import DDPMSampler as Sampler
        elif sampler_type == "dpm_solver":
            from samplers.dpm_solver_sampler import DPMSolverSampler as Sampler
        else:
            raise NotImplementedError("Unknown sampler type [%s]!!!" % sampler_type)
        return Sampler(args, config)

    def get_model_class(self, network: str):
        # Architectures for DDPM
        if network == "ddpm.unet":
            from models.ddpm_arch.unet import UNet as Model
        elif network == "ddpm.fused_unet":
            from models.ddpm_arch.fused_unet import FusedUNet as Model
        elif network == "ddpm.sige_fused_unet":
            from models.ddpm_arch.sige_fused_unet import SIGEFusedUNet as Model
        else:
            raise NotImplementedError("Unknown network [%s]!!!" % network)
        return Model

    def build_model(self):
        args, config = self.args, self.config
        Model = self.get_model_class(config.model.network)
        model = Model(args, config)
        model = model.to(self.device)

        if config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        return model, ema_helper

    def restore_checkpoint(self, model: nn.Module, ema_helper: Optional[EMAHelper]):
        if isinstance(model, nn.DataParallel):
            model = model.module
        args = self.args
        if args.restore_from is not None:
            states = torch.load(args.restore_from, map_location="cpu")
            model.load_state_dict(states["model"])
            if ema_helper is not None:
                if "ema" not in states:
                    ema_helper.register(model)
                else:
                    ema_helper.load_state_dict(states["ema"])
        return model, ema_helper

    def get_sampling_sequence(self, noise_level=None):
        config = self.config
        if noise_level is None:
            noise_level = self.config.sampling.total_steps

        skip_type = config.sampling.skip_type
        timesteps = config.sampling.sample_steps

        if skip_type == "uniform":
            skip = noise_level // timesteps
            seq = range(0, noise_level, skip)
        elif skip_type == "quad":
            seq = np.linspace(0, np.sqrt(noise_level * 0.8), timesteps - 1) ** 2
            seq = [int(s) for s in list(seq)]
            seq.append(noise_level)
        else:
            raise NotImplementedError("Unknown skip type [%s]!!!" % skip_type)
        return seq

    def sample_image(self, x, model, **kwargs):
        noise_level = kwargs.pop("noise_level", None)
        seq = self.get_sampling_sequence(noise_level)
        return self.sampler.denoising_steps(x, model, seq, **kwargs)

    def preprocess(self, original_img, edited_img, model: nn.Module, mode="full"):
        args, config = self.args, self.config
        set_seed(args.seed)
        if self.noise is None:
            self.noise = torch.randn(original_img.shape, device=self.device)
        e = self.noise

        original_img = data_transform(config, original_img)
        edited_img = data_transform(config, edited_img)

        eps = config.sampling.eps
        difference_mask = compute_difference_mask(original_img, edited_img, eps=eps)
        difference_mask = dilate_mask(difference_mask, config.sampling.mask_dilate_radius)

        if isinstance(model, SIGEModel) and mode != "full":
            masks = downsample_mask(difference_mask, config.data.image_size // (2 ** (len(config.model.ch_mult) - 1)))
            model.set_masks(masks)

        x0s = edited_img
        es = e
        return x0s, es, difference_mask

    def generate(self, original_img: torch.Tensor, edited_img: torch.Tensor, mode="full", sparse_update=False):
        args, config = self.args, self.config
        model = self.model
        seq = self.seq

        if isinstance(model, SIGEModel):
            model.set_mode(mode)
            model.set_sparse_update(sparse_update)

        with torch.no_grad():
            x0s, es, difference_mask = self.preprocess(original_img, edited_img, model, mode=mode)

            if self.is_sige_model() and difference_mask.sum() == 0 and mode != "full":
                return edited_img.cpu()

            print("Edit Ratio %.2f%%" % (100 * float(difference_mask.sum() / difference_mask.numel())))

            ts = torch.full((x0s.size(0),), seq[-1], device=x0s.device, dtype=torch.long)
            xts = self.sampler.get_xt_from_x0(x0s, ts, es)
            gt_x0 = x0s
            gt_e = es
            generated_x0s = self.sample_image(
                xts,
                model,
                noise_level=config.sampling.noise_level,
                difference_mask=difference_mask,
                gt_x0=gt_x0,
                gt_e=gt_e,
            )
            generated_x0 = inverse_data_transform(config, generated_x0s.cpu())
        return generated_x0

    def is_sige_model(self):
        return isinstance(self.model, SIGEModel)
