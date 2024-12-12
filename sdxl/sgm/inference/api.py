import pathlib
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

from omegaconf import OmegaConf

from sgm.inference.helpers import (Img2ImgDiscretizationWrapper, do_img2img,
                                   do_sample, do_inpaint, do_sdedit)
from sgm.modules.diffusionmodules.sampling import (DPMPP2MSampler,
                                                   SubstepSampler,)
from sgm.util import load_model_from_config

from sige.utils import downsample_mask
import numpy as np
import torch


class ModelArchitecture(str, Enum):
    SDXL_V1_BASE = "stable-diffusion-xl-v1-base"
    SDXL_SIGE_V1_BASE = "sige-stable-diffusion-xl-v1-base"
    SDXL_V1_REFINER = "stable-diffusion-xl-v1-refiner"
    SDXL_SIGE_V1_REFINER = "sige-stable-diffusion-xl-v1-refiner"
    SDXL_TURBO = "stable-diffusion-xl-turbo"
    SDXL_SIGE_TURBO = "sige-stable-diffusion-xl-turbo"


class Sampler(str, Enum):
    DPMPP2M = "DPMPP2MSampler"
    TURBO_SAMPLER = "TurboSampler"


class Discretization(str, Enum):
    LEGACY_DDPM = "LegacyDDPMDiscretization"
    EDM = "EDMDiscretization"


class Guider(str, Enum):
    VANILLA = "VanillaCFG"
    IDENTITY = "IdentityGuider"


class Thresholder(str, Enum):
    NONE = "None"


@dataclass
class SamplingParams:
    width: int = 1024
    height: int = 1024
    steps: int = 50
    sampler: Sampler = Sampler.DPMPP2M
    discretization: Discretization = Discretization.LEGACY_DDPM
    guider: Guider = Guider.VANILLA
    thresholder: Thresholder = Thresholder.NONE
    scale: float = 6.0
    aesthetic_score: float = 5.0
    negative_aesthetic_score: float = 5.0
    img2img_strength: float = 1.0
    orig_width: int = 1024
    orig_height: int = 1024
    crop_coords_top: int = 0
    crop_coords_left: int = 0
    sigma_min: float = 0.0292
    sigma_max: float = 14.6146
    rho: float = 3.0
    s_churn: float = 0.0
    s_tmin: float = 0.0
    s_tmax: float = 999.0
    s_noise: float = 1.0
    eta: float = 1.0
    order: int = 4


@dataclass
class SamplingSpec:
    width: int
    height: int
    channels: int
    factor: int
    is_legacy: bool
    config: str
    ckpt: str
    is_guided: bool


model_specs = {
    ModelArchitecture.SDXL_V1_BASE: SamplingSpec(
        height=1024,
        width=1024,
        channels=4,
        factor=8,
        is_legacy=False,
        config="original.yaml",
        ckpt="sd_xl_base_1.0.safetensors",
        is_guided=True,
    ),
    ModelArchitecture.SDXL_SIGE_V1_BASE: SamplingSpec(
        height=1024,
        width=1024,
        channels=4,
        factor=8,
        is_legacy=False,
        config="sige.yaml",
        ckpt="sd_xl_base_1.0.safetensors",
        is_guided=True,
    ),
    ModelArchitecture.SDXL_V1_REFINER: SamplingSpec(
        height=1024,
        width=1024,
        channels=4,
        factor=8,
        is_legacy=True,
        config="refiner.yaml",
        ckpt="sd_xl_refiner_1.0.safetensors",
        is_guided=True,
    ),
    ModelArchitecture.SDXL_SIGE_V1_REFINER: SamplingSpec(
        height=1024,
        width=1024,
        channels=4,
        factor=8,
        is_legacy=True,
        config="sige-refiner.yaml",
        ckpt="sd_xl_refiner_1.0.safetensors",
        is_guided=True,
    ),
    ModelArchitecture.SDXL_TURBO: SamplingSpec(
        height=1024,
        width=1024,
        channels=4,
        factor=8,
        is_legacy=False,
        config="original.yaml",
        ckpt="sd_xl_turbo_1.0.safetensors",
        is_guided=True,
    ),
    ModelArchitecture.SDXL_SIGE_TURBO: SamplingSpec(
        height=1024,
        width=1024,
        channels=4,
        factor=8,
        is_legacy=False,
        config="sige.yaml",
        ckpt="sd_xl_turbo_1.0.safetensors",
        is_guided=True,
    ),
}


class SamplingPipeline:
    def __init__(
        self,
        model_id: ModelArchitecture,
        model_path="checkpoints",
        config_path="configs/inference",
        device="cuda",
        mask_path=None,
        args=None,
        use_fp16=True,
    ) -> None:
        if model_id not in model_specs:
            raise ValueError(f"Model {model_id} not supported")
        self.model_id = model_id
        self.specs = model_specs[self.model_id]
        self.config = str(pathlib.Path(config_path, self.specs.config))
        self.ckpt = str(pathlib.Path(model_path, self.specs.ckpt))
        self.device = device
        self.args = args
        self.model = self._load_model(device=device, use_fp16=use_fp16)


    def _load_model(self, device="cuda", use_fp16=True):
        config = OmegaConf.load(self.config)
        if self.args.mode == 'profile_unet': config['model']['params']['network_config']['params']['use_checkpoint'] = False
        if self.args.mode in ['profile_encoder', 'profile_decoder']: 
            attn_type = 'sige' if self.args.run_type == 'sige' else 'vanilla'
            config['model']['params']['first_stage_config']['params']['ddconfig']['attn_type'] = attn_type

        model = load_model_from_config(config, self.ckpt)
        if model is None:
            raise ValueError(f"Model {self.model_id} could not be loaded")
        model.to(device)
        if use_fp16:
            model.conditioner.half()
            model.model.half()
        return model

    def text_to_image(
        self,
        params: SamplingParams,
        prompt: str,
        negative_prompt: str = "",
        samples: int = 1,
        return_latents: bool = False,
    ):
        sampler = get_sampler_config(params)
        value_dict = asdict(params)
        value_dict["prompt"] = prompt
        value_dict["negative_prompt"] = negative_prompt
        value_dict["target_width"] = params.width
        value_dict["target_height"] = params.height
        return do_sample(
            self.model,
            sampler,
            value_dict,
            samples,
            params.height,
            params.width,
            self.specs.channels,
            self.specs.factor,
            force_uc_zero_embeddings=["txt"] if not self.specs.is_legacy else [],
            return_latents=return_latents,
            filter=None,
        )

    def inpaint(
        self,
        params: SamplingParams,
        image,
        prompt: str,
        mask,
        conv_masks,
        negative_prompt: str = "",
        samples: int = 1,
        return_latents: bool = False,
        skip_encode = False,
    ):
        sampler = get_sampler_config(params)

        if params.img2img_strength < 1.0:
            sampler.discretization = Img2ImgDiscretizationWrapper(
                sampler.discretization,
                strength=params.img2img_strength,
            )
        height, width = image.shape[2], image.shape[3]
        value_dict = asdict(params)
        value_dict["prompt"] = prompt
        value_dict["negative_prompt"] = negative_prompt
        value_dict["target_width"] = width
        value_dict["target_height"] = height
        return do_inpaint(
            image,
            self.model,
            sampler,
            value_dict,
            samples,
            force_uc_zero_embeddings=["txt"] if not self.specs.is_legacy else [],
            return_latents=return_latents,
            filter=None,
            mask=mask,
            conv_masks=conv_masks,
            args = self.args,
            skip_encode=skip_encode,
        )

    def sdedit(
        self,
        params: SamplingParams,
        edited_image,
        init_image,
        masks,
        prompt: str,
        negative_prompt: str = "",
        samples: int = 1,
        return_latents: bool = False,
        is_sige_model=False,
        difference_mask=None,
        skip_encode=False,
    ):
        sampler = get_sampler_config(params)

        if params.img2img_strength < 1.0:
            sampler.discretization = Img2ImgDiscretizationWrapper(
                sampler.discretization,
                strength=params.img2img_strength,
            )
        height, width = edited_image.shape[2], edited_image.shape[3]
        value_dict = asdict(params)
        value_dict["prompt"] = prompt
        value_dict["negative_prompt"] = negative_prompt
        value_dict["target_width"] = width
        value_dict["target_height"] = height
        return do_sdedit(
            edited_image,
            self.model,
            sampler,
            value_dict,
            samples,
            force_uc_zero_embeddings=["txt"] if not self.specs.is_legacy else [],
            return_latents=return_latents,
            filter=None,
            init_img=init_image,
            masks=masks,
            is_sige_model=is_sige_model,
            difference_mask=difference_mask,
            args = self.args,
            skip_encode=skip_encode,
        )

    def image_to_image(
        self,
        params: SamplingParams,
        image,
        prompt: str,
        negative_prompt: str = "",
        samples: int = 1,
        return_latents: bool = False,
        difference_mask=None,
    ):
        sampler = get_sampler_config(params)

        if params.img2img_strength < 1.0:
            sampler.discretization = Img2ImgDiscretizationWrapper(
                sampler.discretization,
                strength=params.img2img_strength,
            )
        height, width = image.shape[2], image.shape[3]
        value_dict = asdict(params)
        value_dict["prompt"] = prompt
        value_dict["negative_prompt"] = negative_prompt
        value_dict["target_width"] = width
        value_dict["target_height"] = height
        return do_img2img(
            image,
            self.model,
            sampler,
            value_dict,
            samples,
            force_uc_zero_embeddings=["txt"] if not self.specs.is_legacy else [],
            return_latents=return_latents,
            filter=None,
            masks=self.masks,
            conv_masks=self.conv_masks,
            shape=self.shape,
            difference_mask=difference_mask,
        )

    def refiner(
        self,
        params: SamplingParams,
        image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        samples: int = 1,
        return_latents: bool = False,
    ):
        sampler = get_sampler_config(params)
        value_dict = {
            "orig_width": image.shape[3] * 8,
            "orig_height": image.shape[2] * 8,
            "target_width": image.shape[3] * 8,
            "target_height": image.shape[2] * 8,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "crop_coords_top": 0,
            "crop_coords_left": 0,
            "aesthetic_score": 6.0,
            "negative_aesthetic_score": 2.5,
        }

        return do_img2img(
            image,
            self.model,
            sampler,
            value_dict,
            samples,
            skip_encode=True,
            return_latents=return_latents,
            filter=None,
        )


def get_guider_config(params: SamplingParams):
    if params.guider == Guider.IDENTITY:
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"
        }
    elif params.guider == Guider.VANILLA:
        scale = params.scale

        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {"scale": scale},
        }
    else:
        raise NotImplementedError
    return guider_config


def get_discretization_config(params: SamplingParams):
    if params.discretization == Discretization.LEGACY_DDPM:
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
        }
    elif params.discretization == Discretization.EDM:
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {
                "sigma_min": params.sigma_min,
                "sigma_max": params.sigma_max,
                "rho": params.rho,
            },
        }
    else:
        raise ValueError(f"unknown discretization {params.discretization}")
    return discretization_config


def get_sampler_config(params: SamplingParams):
    discretization_config = get_discretization_config(params)
    guider_config = get_guider_config(params)
    sampler = None

    if params.sampler == Sampler.DPMPP2M:
        return DPMPP2MSampler(
            num_steps=params.steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    if params.sampler == Sampler.TURBO_SAMPLER:
        return SubstepSampler(
            n_sample_steps=params.steps,
            num_steps=1000,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )

    raise ValueError(f"unknown sampler {params.sampler}!")
