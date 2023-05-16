from typing import Optional

import torch
from torch import nn
from tqdm import tqdm


class BaseSampler:
    def __init__(self, args, config) -> None:
        self.args, self.config = args, config
        self.device = config.device

    def denoising_steps(self, x: torch.Tensor, model: nn.Module, seq, **kwargs):
        seq_next = [-1] + list(seq[:-1])
        sampling_tqdm = tqdm(zip(reversed(seq), reversed(seq_next)), total=len(seq), desc="Sampling   ")

        with torch.no_grad():
            for step, (i, j) in enumerate(sampling_tqdm):
                x = self.denoising_step(model, x, i, j, step=step, **kwargs)
        return x

    def denoising_step(self, model: nn.Module, x: torch.Tensor, i: int, j: int, step: int, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def get_xt_from_x0(self, x0: torch.Tensor, t: torch.Tensor, e: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def post_process(self, x: torch.Tensor, t, **kwargs):
        difference_mask = kwargs.pop("difference_mask", None)
        if difference_mask.dim() == 2:
            difference_mask = difference_mask.unsqueeze(0)

        gt_x0 = kwargs.pop("gt_x0", None)
        gt_e = kwargs.pop("gt_e", None)
        gt_xt = None if (gt_x0 is None or gt_e is None) else self.get_xt_from_x0(gt_x0, t[:1], gt_e)
        if x.size(0) == 2:
            assert gt_xt is not None
            x[:1] = gt_xt
        if difference_mask is not None:
            x[-1] = gt_xt * (~difference_mask) + x[-1] * difference_mask
        return x
