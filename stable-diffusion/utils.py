import random
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from imwatermark import WatermarkEncoder
from PIL import Image
from torch.backends import cudnn
from tqdm import tqdm
from transformers import AutoFeatureExtractor

from ldm.util import instantiate_from_config

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def load_model_from_config(config, weight_path: str, verbose: bool = False):
    print(f"Loading model from {weight_path}")
    pl_sd = torch.load(weight_path, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()
    model.eval()
    return model


def put_watermark(img: Image, wm_encoder: Optional[WatermarkEncoder] = None) -> Image:
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, "dwtDct")
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_img(path: str, size: Optional[Union[int, Tuple[int, int]]] = None):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    if size is None:
        w, h = w - w % 32, h - h % 32  # resize to integer multiple of 32
    else:
        if isinstance(size, int):
            size = size - size % 32
            w, h = size, size
        else:
            w, h = size
            w, h = w - w % 32, h - h % 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def set_seed(seed: Optional[int] = None):
    cudnn.benchmark = True  # if benchmark=True, deterministic will be False
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def mytqdm(iterable, **kwargs):
    position = kwargs.get("position", None)
    if position is None or position >= 0:
        return tqdm(iterable, **kwargs)
    else:
        return iterable
