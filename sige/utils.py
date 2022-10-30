from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F


def reduce_mask(
    mask: torch.Tensor,
    block_size: Optional[Union[int, Tuple[int, int]]],
    stride: Optional[Union[int, Tuple[int, int]]],
    padding: Optional[Union[int, Tuple[int, int]]],
    verbose: bool = False,
) -> Optional[torch.Tensor]:
    if block_size is None or stride is None or padding is None:
        return None
    else:
        if isinstance(block_size, int):
            block_size = (block_size, block_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(stride, int):
            stride = (stride, stride)
        H, W = mask.shape
        # Max Pooling only supports float tensor
        mask = mask.view(1, 1, H, W).to(torch.float32)
        mask = F.pad(mask, (padding[1], block_size[1], padding[0], block_size[0]))
        mask_pooled = F.max_pool2d(mask, block_size, stride)
        mask_pooled = mask_pooled[0, 0] > 0.5
        active_indices = torch.nonzero(mask_pooled)
        active_indices[:, 0] = stride[0] * active_indices[:, 0] - padding[0]
        active_indices[:, 1] = stride[1] * active_indices[:, 1] - padding[1]
        if verbose:
            num_active = active_indices.shape[0]
            total = mask_pooled.numel()
            print("Block Sparsity: %d/%d=%.2f%%" % (num_active, total, 100 * num_active / total))
        return active_indices.to(torch.int32).contiguous()


def dilate_mask(
    mask: Union[torch.Tensor, np.ndarray], dilation: Union[int, Tuple[int, int]]  # [C, H, W] or [H, W]
) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    if dilation[0] <= 0 and dilation[1] <= 0:
        return mask

    if isinstance(mask, torch.Tensor):
        ret = mask.clone()
    else:
        assert isinstance(mask, np.ndarray)
        ret = mask.copy()

    if len(ret.shape) == 2:
        for i in range(1, dilation[0] + 1):
            ret[:-i] |= mask[i:]
            ret[i:] |= mask[:-i]
        for i in range(1, dilation[1] + 1):
            ret[:, :-i] |= mask[:, i:]
            ret[:, i:] |= mask[:, :-i]
    elif len(ret.shape) == 3:
        for i in range(1, dilation + 1):
            ret[:, :-i] |= mask[:, i:]
            ret[:, i:] |= mask[:, :-i]
        for i in range(1, dilation[1] + 1):
            ret[:, :, :-i] |= mask[:, :, i:]
            ret[:, :, i:] |= mask[:, :, :-i]
    else:
        raise NotImplementedError("Unknown mask dimension [%d]!!!" % mask.dim())
    return ret


def compute_difference_mask(tensor1: torch.Tensor, tensor2: torch.Tensor, eps: float = 2e-2) -> torch.Tensor:
    difference = torch.abs(tensor1 - tensor2)
    mask = difference > eps
    if mask.dim() == 2:  # [H, W]
        return mask
    elif mask.dim() == 3:  # [C, H, W]
        return torch.any(mask, 0)
    elif mask.dim() == 4:  # [B, C, H, W]
        assert mask.shape[0] == 1
        return torch.any(mask[0], 0)
    else:
        raise NotImplementedError("Unknown mask dimension [%d]!!!" % mask.dim())


def downsample_mask(
    mask: torch.Tensor,
    min_res: Union[int, Tuple[int, int]] = 4,
    dilation: Union[int, Tuple[int, int]] = 1,
    threshold: float = 0.3,
    eps: float = 1e-3,
) -> Dict[Tuple[int, int], torch.Tensor]:
    assert mask.dim() == 2
    H, W = mask.shape
    if isinstance(min_res, int):
        min_h = min_res
        min_w = min_res
    else:
        min_h, min_w = min_res
    h = H
    w = W

    masks = {}
    interpolated_mask = mask.view(1, 1, H, W).float()
    while True:
        t = min(threshold, interpolated_mask.max() - eps)
        sparsity_mask = interpolated_mask[0, 0] > t
        sparsity_mask = dilate_mask(sparsity_mask, dilation)
        masks[(h, w)] = sparsity_mask
        h //= 2
        w //= 2
        if h < min_h and w < min_w:
            break
        else:
            interpolated_mask = F.interpolate(interpolated_mask, (h, w), mode="bilinear", align_corners=False)
    return masks
