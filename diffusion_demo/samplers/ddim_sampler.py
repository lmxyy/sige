import torch
from torch import nn

from samplers.ddim_ddpm_sampler import DDIMDDPMSampler, compute_alpha


class DDIMSampler(DDIMDDPMSampler):
    def __init__(self, args, config):
        super(DDIMSampler, self).__init__(args, config)

    def denoising_step(self, model: nn.Module, x: torch.Tensor, i: int, j: int, step: int, **kwargs) -> torch.Tensor:
        n = x.size(0)
        betas = self.betas

        t = torch.full((n,), fill_value=i, device=x.device)
        next_t = torch.full((n,), fill_value=j, device=x.device)
        at = compute_alpha(betas, t.long())
        atm1 = compute_alpha(betas, next_t.long())

        eta = self.config.sampling.eta
        x0_t, et = self.model_step(model, x, t, at, step=step)
        c1 = eta * ((1 - at / atm1) * (1 - atm1) / (1 - at)).sqrt()
        c2 = ((1 - atm1) - c1**2).sqrt()
        xt_next = atm1.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et

        xt_next = self.post_process(xt_next, next_t, **kwargs)
        return xt_next
