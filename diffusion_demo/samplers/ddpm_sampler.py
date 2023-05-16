import torch
from torch import nn

from samplers.ddim_ddpm_sampler import DDIMDDPMSampler, compute_alpha


class DDPMSampler(DDIMDDPMSampler):
    def __init__(self, args, config):
        super(DDPMSampler, self).__init__(args, config)

    def denoising_step(self, model: nn.Module, x: torch.Tensor, i: int, j: int, step: int, **kwargs) -> torch.Tensor:
        n = x.size(0)
        betas = self.betas

        t = torch.full((n,), fill_value=i, device=x.device)
        next_t = torch.full((n,), fill_value=j, device=x.device)
        at = compute_alpha(betas, t.long())
        atm1 = compute_alpha(betas, next_t.long())

        beta_t = 1 - at / atm1
        x0_from_e, e = self.model_step(model, x, t, at, step=step)
        x0_from_e = torch.clamp(x0_from_e, -1, 1)
        mean_eps = ((atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x) / (1.0 - at)
        mean = mean_eps
        noise = torch.randn_like(x0_from_e)
        mask = 1 - (t == 0).float()
        mask = mask.view(-1, 1, 1, 1)
        logvar = beta_t.log()
        sample = mean + mask * torch.exp(0.5 * logvar) * noise

        sample = self.post_process(sample, next_t, **kwargs)
        return sample
