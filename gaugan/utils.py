import os
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from colorize import Colorize


def decode_config(config_str: str):
    channels = config_str.split("_")
    channels = [int(c) for c in channels]
    return {"channels": channels}


def load_network(net: nn.Module, path: str, verbose: bool = False) -> nn.Module:
    old_state_dict = net.state_dict()
    new_state_dict = torch.load(path)
    state_dict = {}
    for k, v in old_state_dict.items():
        vv = new_state_dict[k]
        if v.shape != vv.shape:
            assert v.dim() == vv.dim() == 1
            assert "param_free_norm" in k
            state_dict[k] = vv[: v.shape[0]]
        else:
            state_dict[k] = vv
    net.load_state_dict(state_dict)
    return net


def mytqdm(iterable, **kwargs):
    position = kwargs.get("position", None)
    if position is None or position >= 0:
        return tqdm(iterable, **kwargs)
    else:
        return iterable


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image, normalize=normalize)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8, tile=False):
    if label_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(label_tensor.size(0)):
            one_image = label_tensor[b]
            one_image_np = tensor2label(one_image, n_label, imtype)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)

    if label_tensor.dim() == 1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result


def save_image(image_numpy, image_path, create_dir=True):
    if create_dir:
        os.makedirs(os.path.dirname(os.path.abspath(image_path)), exist_ok=True)
    if len(image_numpy.shape) == 4:
        image_numpy = image_numpy[0]
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = Image.fromarray(image_numpy)

    # save to png
    image_pil.save(image_path.replace(".jpg", ".png"))


def save_visuals(opt, visuals: Dict[str, torch.Tensor], name: str):
    for k, v in visuals.items():
        save_dir = os.path.join(opt.save_dir, k)
        save_path = os.path.join(save_dir, "%s.png" % name)
        if k in ("original_label", "edited_label"):
            t = tensor2label(v, opt.input_nc + 1)
        else:
            t = tensor2im(v)
        save_image(t, save_path)
