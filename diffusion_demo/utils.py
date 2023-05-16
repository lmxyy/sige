import random
from argparse import Namespace
from typing import Optional

import numpy as np
import torch
from torch.backends import cudnn


def data_transform(config, x: torch.Tensor):
    if config.data.rescaled:
        x = 2 * x - 1.0
    return x


def inverse_data_transform(config, x: torch.Tensor):
    if config.data.rescaled:
        x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)


def override_config(hparams_strings: str, config):
    hparams_strings = hparams_strings.strip()

    def my_getattr(attributes):
        ret = config
        for a in attributes:
            if not hasattr(ret, a):
                setattr(ret, a, Namespace())
            ret = getattr(ret, a)
        return ret

    strings = hparams_strings.split(" ")
    for string in strings:
        string = string.strip()
        if "=" in string:
            key, value = string.split("=")
            keys = key.split(".")
            setattr(my_getattr(keys[:-1]), keys[-1], parse_value(value, my_getattr(keys)))


def parse_value(value: str, original_value=None):
    def is_type(v: str, t):
        try:
            t(v)
        except:
            return False
        else:
            return True

    value = value.strip()
    if value.lower() == "none":
        ret = None
        if isinstance(original_value, (list, tuple)):
            ret = [ret]
    elif value.lower() == "true":
        ret = True
        if isinstance(original_value, (list, tuple)):
            ret = [ret]
    elif value.lower() == "false":
        ret = False
        if isinstance(original_value, (list, tuple)):
            ret = [ret]
    elif is_type(value, int):
        ret = int(value)
        if isinstance(original_value, (list, tuple)):
            ret = [ret]
    elif is_type(value, float):
        ret = float(value)
        if isinstance(original_value, (list, tuple)):
            ret = [ret]
    elif "," in value:
        values = value.split(",")
        ret = [parse_value(v, None) for v in values]
    else:
        t = type(original_value)
        ret = t(value)
    return ret


def set_seed(seed):
    cudnn.benchmark = True  # if benchmark=True, deterministic will be False
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)
