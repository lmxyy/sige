import os
import time
import warnings
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision import utils as tvu

from download_helper import get_ckpt_path
from datasets import data_transform, get_dataset, inverse_data_transform
from models.ema import EMAHelper
from samplers.base_sampler import BaseSampler
from sige.nn import SIGEModel
from sige.utils import compute_difference_mask, dilate_mask, downsample_mask
from utils import mytqdm, set_seed


class Runner:
    def __init__(self, args, config):
        self.args, self.config = args, config
        self.device = config.device
        self.sampler = self.get_sampler()

    def get_sampler(self) -> BaseSampler:
        args, config = self.args, self.config
        sampler_type = config.sampling.sampler_type
        if sampler_type == "ddim":
            from samplers.ddim_sampler import DDIMSampler as Sampler
        elif sampler_type == "ddpm":
            from samplers.ddpm_sampler import DDPMSampler as Sampler
        elif sampler_type == "pd":
            from samplers.pd_sampler import PDSampler as Sampler
        else:
            raise NotImplementedError("Unknown sampler type [%s]!!!" % sampler_type)
        return Sampler(args, config)

    def get_model_class(self, network: str):
        # Architectures for DDPM/DDIM Sampler
        if network == "ddim.unet":
            from models.ddim_arch.unet import UNet as Model
        elif network == "ddim.fused_unet":
            from models.ddim_arch.fused_unet import FusedUNet as Model
        elif network == "ddim.sige_fused_unet":
            from models.ddim_arch.sige_fused_unet import SIGEFusedUNet as Model
        # Architectures for Progressive Distillation Sammpler
        elif network == "pd.unet":
            from models.pd_arch.unet import UNet as Model
        elif network == "pd.sige_unet":
            from models.pd_arch.sige_unet import SIGEUNet as Model
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
            states = torch.load(args.restore_from)
            model.load_state_dict(states["model"])
            if ema_helper is not None:
                if "ema" not in states:
                    ema_helper.register(model)
                else:
                    ema_helper.load_state_dict(states["ema"])
        return model, ema_helper

    def run(self):
        args, config = self.args, self.config
        model, ema_helper = self.build_model()
        if args.use_pretrained:
            pretrained_path = get_ckpt_path(config, tool=args.download_tool)
            if args.restore_from is not None:
                warnings.warn("The model path will be overriden to [%s]!!!" % pretrained_path)
            args.restore_from = pretrained_path
        self.restore_checkpoint(model, ema_helper=ema_helper)
        if ema_helper is not None:
            ema_helper.ema(model)

        model.eval()

        dataset = get_dataset(args, config)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.data.num_workers)

        if args.mode == "profile":
            self.profile(model, dataloader)
        elif args.mode == "generate":
            self.generate(model, dataloader)
        else:
            raise NotImplementedError("Unknown test mode [%s]!!!" % args.mode)

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

    def get_nxe(self, x, e: Optional[torch.Tensor] = None):
        config = self.config
        n = x.size(0)
        x = x.to(self.device)[:, : config.model.in_ch]  # sometimes the input have more channels
        x = data_transform(config, x)
        if e is None:
            e = torch.randn_like(x)
        else:
            e = e.to(self.device)
            assert e.shape == x.shape
        return n, x, e

    def preprocess(self, batch, model: nn.Module, **kwargs):
        args, config = self.args, self.config
        set_seed(args.seed)  # for reproducibility
        original_img, edited_img, _, names = batch
        _, original_img, e = self.get_nxe(original_img)
        _, edited_img, e = self.get_nxe(edited_img, e=e)

        eps = config.sampling.eps
        difference_mask = compute_difference_mask(original_img, edited_img, eps=eps)
        difference_mask = dilate_mask(difference_mask, config.sampling.mask_dilate_radius)

        if isinstance(model, SIGEModel):
            # pre-run the model to get the shape info and set the mask
            model.set_mode("full")
            model(original_img, torch.zeros(1, device=original_img.device, dtype=torch.float32))
            masks = downsample_mask(difference_mask, config.data.image_size // (2 ** (len(config.model.ch_mult) - 1)))
            model.set_masks(masks)

        if kwargs.pop("verbose", False):
            edit_ratio = float(difference_mask.sum() / difference_mask.numel())
            message = "Image %s: Edit Ratio %.3f%%" % (names[0], 100 * edit_ratio)
            pbar = kwargs.pop("pbar", None)
            if pbar is None:
                pbar.write(message + "\n")
            else:
                print(message)

        x0s = torch.cat([original_img, edited_img], dim=0)
        es = torch.cat([e, e], dim=0)
        return x0s, es, difference_mask, names

    def generate(self, model: nn.Module, dataloader: DataLoader):
        args, config = self.args, self.config

        save_dir = args.save_dir
        if save_dir is None:
            warnings.warn("No save directory specified, no images will be saved.")

        noise_level = config.sampling.noise_level
        with torch.no_grad():
            pbar_dataloader = mytqdm(dataloader, position=0, desc="Batch      ", leave=False)
            for batch in pbar_dataloader:
                x0s, es, difference_mask, names = self.preprocess(batch, model, pbar=pbar_dataloader)
                pbar_dataloader.write(
                    "Image %s: Edit Ratio %.2f%%"
                    % (names[0], 100 * float(difference_mask.sum() / difference_mask.numel()))
                )

                seq = self.get_sampling_sequence(noise_level)
                ts = torch.full((x0s.size(0),), seq[-1], device=x0s.device, dtype=torch.long)
                xts = self.sampler.get_xt_from_x0(x0s, ts, es)
                gt_x0 = x0s[:1]
                gt_e = es[:1]
                if not isinstance(model, SIGEModel):
                    xts = xts[1:]
                generated_x0s = self.sample_image(
                    xts, model, noise_level=noise_level, difference_mask=difference_mask, gt_x0=gt_x0, gt_e=gt_e
                )
                generated_x0 = inverse_data_transform(config, generated_x0s[-1].cpu())
                if save_dir is not None:
                    os.makedirs(os.path.abspath(save_dir), exist_ok=True)
                    name = names[0]
                    filename = name + ".png"
                    tvu.save_image(generated_x0, os.path.join(save_dir, filename))

    def profile(self, model: nn.Module, dataloader: DataLoader):
        args = self.args
        with torch.no_grad():
            pbar_dataloader = mytqdm(dataloader, position=0, desc="Batch      ", leave=False)
            for batch in pbar_dataloader:
                x0s, _, difference_mask, names = self.preprocess(batch, model, pbar=pbar_dataloader)
                dummy_inputs = (x0s[:1], torch.zeros(1, device=x0s.device, dtype=torch.float32))
                if isinstance(model, SIGEModel):
                    model.set_mode("sparse")

                for _ in mytqdm(range(args.warmup_times), position=1, desc="Warmup     ", leave=False):
                    model(*dummy_inputs)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                start_time = time.time()
                for _ in mytqdm(range(args.test_times), position=1, desc="Measure    ", leave=False):
                    model(*dummy_inputs)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                cost_time = time.time() - start_time

                if isinstance(model, SIGEModel):
                    model.set_mode("profile")
                macs = profile_macs(model, dummy_inputs)
                pbar_dataloader.write(
                    "Image %s: Sparsity %.2f%%    MACs %.3fG    Cost Time %.3fs    Avg Time %.3fms"
                    % (
                        names[0],
                        100 * float(difference_mask.sum() / difference_mask.numel()),
                        macs / 1e9,
                        cost_time,
                        cost_time / args.test_times * 1000,
                    )
                )
