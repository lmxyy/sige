import math
from typing import Optional

import torch
from torch import nn

from samplers.base_sampler import BaseSampler
from sige.nn import SIGEModel


def logsnr_schedule(t, logsnr_min=-20.0, logsnr_max=20.0):
    b = math.atan(math.exp(-0.5 * logsnr_max))
    a = math.atan(math.exp(-0.5 * logsnr_min)) - b
    return -2.0 * torch.log(torch.tan(a * t + b))


def diffusion_forward(x, logsnr):
    """q(z_t | x)."""
    return {
        "mean": x * torch.sqrt(torch.sigmoid(logsnr)),
        "std": torch.sqrt(torch.sigmoid(-logsnr)),
        "var": torch.sigmoid(-logsnr),
        "logvar": torch.log(torch.sigmoid(-logsnr)),
    }


def predict_x_from_eps(z, eps, logsnr):
    logsnr = logsnr[:, None, None, None]
    return torch.sqrt(1.0 + torch.exp(-logsnr)) * (z - eps / torch.sqrt(1.0 + torch.exp(logsnr)))


def predict_eps_from_x(z, x, logsnr):
    """eps = (z - alpha*x)/sigma."""
    logsnr = logsnr[:, None, None, None]
    return torch.sqrt(1.0 + torch.exp(logsnr)) * (z - x / torch.sqrt(1.0 + torch.exp(-logsnr)))


def predict_v_from_x_and_eps(x, eps, logsnr):
    logsnr = logsnr[:, None, None, None]
    alpha_t = torch.sqrt(torch.sigmoid(logsnr))
    sigma_t = torch.sqrt(torch.sigmoid(-logsnr))
    return alpha_t * eps - sigma_t * x


def run_model(model: nn.Module, z, logsnr):
    if isinstance(model, SIGEModel):
        assert z.size(0) == 2
        model.set_mode("full")
        output0 = model(z[:1], logsnr[:1])
        model.set_mode("sparse")
        output1 = model(z[1:], logsnr[:1])
        model_output = torch.cat([output0, output1], dim=0)
    else:
        assert z.size(0) == 1
        model_output = model(z, logsnr)
    _model_x, _model_eps = model_output[:, :3], model_output[:, 3:]
    model_x_eps = predict_x_from_eps(z=z, eps=_model_eps, logsnr=logsnr)
    wx = torch.sigmoid(-logsnr)
    wx = wx.view(-1, 1, 1, 1)
    model_x = wx * _model_x + (1.0 - wx) * model_x_eps
    model_x = torch.clip(model_x, -1.0, 1.0)
    model_eps = predict_eps_from_x(z=z, x=model_x, logsnr=logsnr)
    return {"model_x": model_x, "model_eps": model_eps}


class PDSampler(BaseSampler):
    def __init__(self, args, config):
        super(PDSampler, self).__init__(args, config)

    def get_xt_from_x0(self, x0: torch.Tensor, t: torch.Tensor, e: Optional[torch.Tensor] = None) -> torch.Tensor:
        config = self.config
        if e is None:
            e = torch.randn_like(x0)
        u = (t + 1) / config.sampling.total_steps
        logsnr = logsnr_schedule(u)
        z_dict = diffusion_forward(x=x0, logsnr=logsnr.view(-1, 1, 1, 1))
        xt = z_dict["mean"] + z_dict["std"] * e
        return xt

    def denoising_step(self, model: nn.Module, x: torch.Tensor, i: int, j: int, **kwargs) -> torch.Tensor:
        config = self.config
        total_steps = config.sampling.total_steps

        n = x.size(0)
        t = torch.full((n,), fill_value=i, device=x.device)
        next_t = torch.full((n,), fill_value=j, device=x.device)
        logsnr_t = logsnr_schedule((t + 1) / total_steps)
        logsnr_s = logsnr_schedule((next_t + 1) / total_steps)
        model_out = run_model(model=model, z=x, logsnr=logsnr_t)
        x_pred_t = model_out["model_x"]
        eps_pred_t = model_out["model_eps"]
        stdv_s = torch.sqrt(torch.sigmoid(-logsnr_s)).view(-1, 1, 1, 1)
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s)).view(-1, 1, 1, 1)
        z_s_pred = alpha_s * x_pred_t + stdv_s * eps_pred_t
        x = x_pred_t if i == 0 else z_s_pred
        self.post_process(x, next_t, **kwargs)
        return x
