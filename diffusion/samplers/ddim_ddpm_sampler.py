from typing import Optional

import numpy as np
import torch
from torch import nn

from samplers.base_sampler import BaseSampler
from sige.nn import SIGEModel


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = np.linspace(beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class DDIMDDPMSampler(BaseSampler):
    def __init__(self, args, config):
        super(DDIMDDPMSampler, self).__init__(args, config)
        device = self.device

        betas = get_beta_schedule(
            beta_schedule=config.sampling.beta_schedule,
            beta_start=config.sampling.beta_start,
            beta_end=config.sampling.beta_end,
            num_diffusion_timesteps=config.sampling.total_steps,
        )
        betas = torch.from_numpy(betas).float().to(device)
        self.betas = betas
        self.num_timesteps = config.sampling.total_steps

    def get_xt_from_x0(self, x0: torch.Tensor, t: torch.Tensor, e: Optional[torch.Tensor] = None) -> torch.Tensor:
        if e is None:
            e = torch.randn_like(x0)
        a = compute_alpha(self.betas, t.long())
        xt = x0 * a.sqrt() + e * (1 - a).sqrt()
        return xt

    def model_step(self, model: nn.Module, xt, t, at):
        if isinstance(model, SIGEModel):
            assert xt.size(0) == 2
            model.set_mode("full")
            output0 = model(x=xt[:1], t=t[:1].float())
            model.set_mode("sparse")
            output1 = model(x=xt[1:], t=t[1:].float())
            et = torch.cat([output0, output1], dim=0)
            x0 = (xt - et * (1 - at).sqrt()) / at.sqrt()
        else:
            assert xt.size(0) == 1
            et = model(xt, t.float())
            x0 = (xt - et * (1 - at).sqrt()) / at.sqrt()
        return x0, et
