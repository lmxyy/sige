from typing import Optional

import torch
from torch import nn
from tqdm import tqdm

from samplers.base_sampler import BaseSampler
from samplers.ddim_ddpm_sampler import compute_alpha, get_beta_schedule
from sige.nn import SIGEModel
import time


def interpolate_fn(x, xp, yp):
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K),
            torch.tensor(K - 2, device=x.device),
            cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K),
            torch.tensor(K - 2, device=x.device),
            cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


class DPMSolverSampler(BaseSampler):
    def __init__(self, args, config) -> None:
        super().__init__(args, config)
        device = self.device

        betas = get_beta_schedule(
            beta_schedule=config.sampling.beta_schedule,
            beta_start=config.sampling.beta_start,
            beta_end=config.sampling.beta_end,
            num_diffusion_timesteps=config.sampling.total_steps,
        )
        betas = torch.from_numpy(betas).float().to(device)
        self.betas = betas
        self.noise_schedule = NoiseScheduleVP(schedule="discrete", betas=betas)
        self.algorithm_type = config.sampling.algorithm_type
        self.lower_order_final = config.sampling.lower_order_final
        self.solver_type = config.sampling.solver_type
        self.order = config.sampling.order

    def get_xt_from_x0(self, x0: torch.Tensor, t: torch.Tensor, e: Optional[torch.Tensor] = None) -> torch.Tensor:
        if e is None:
            e = torch.randn_like(x0)
        a = compute_alpha(self.betas, t.long())
        xt = x0 * a.sqrt() + e * (1 - a).sqrt()
        return xt

    def convert_t_to_contiguous(self, t: int) -> float:
        return (t + 1) / 1000 + 1 / self.noise_schedule.total_N

    def denoising_steps(self, x: torch.Tensor, model: nn.Module, seq, **kwargs):
        lower_order_final = self.lower_order_final
        solver_type = self.solver_type
        order = self.order

        steps = len(seq)

        sampling_tqdm = tqdm(reversed(seq), total=len(seq), desc="Sampling   ")

        with torch.no_grad():
            for step, t in enumerate(sampling_tqdm):
                t_contiguous = torch.tensor(self.convert_t_to_contiguous(t)).to(x.device).to(torch.float32)
                t = torch.full((x.size(0),), fill_value=t, device=x.device)
                if step == 0:
                    t_prev_list = [t_contiguous]
                    model_prev_list = [self.model_step(model, x, t, step=step)]
                elif step < order:
                    x = self.multistep_dpm_solver_update(
                        x, model_prev_list, t_prev_list, t_contiguous, step, solver_type=solver_type
                    )
                    x = self.post_process(x, t, **kwargs)
                    t_prev_list.append(t_contiguous)
                    model_prev_list.append(self.model_step(model, x, t, step=step))
                else:
                    if lower_order_final and steps < 10:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_dpm_solver_update(
                        x, model_prev_list, t_prev_list, t_contiguous, step_order, solver_type=solver_type
                    )
                    x = self.post_process(x, t, **kwargs)
                    t_prev_list.pop(0)
                    model_prev_list.pop(0)
                    t_prev_list.append(t_contiguous)
                    if step < steps:
                        model_prev_list.append(self.model_step(model, x, t, step=step))

        return x

    def model_step(self, model, xt, t, step: int):
        at = compute_alpha(self.betas, t.long())
        if isinstance(model, SIGEModel):
            model.set_cache_id(step)
        et = model(xt, t.float())

        x0 = (xt - et * (1 - at).sqrt()) / at.sqrt()
        return x0

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type="dpmsolver"):
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def dpm_solver_first_update(self, x, s, t, model_s):
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s
            return x_t
        else:
            phi_1 = torch.expm1(h)
            x_t = torch.exp(log_alpha_t - log_alpha_s) * x - (sigma_t * phi_1) * model_s
            return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
        if solver_type not in ["dpmsolver", "taylor"]:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if solver_type == "dpmsolver":
                x_t = (sigma_t / sigma_prev_0) * x - (alpha_t * phi_1) * model_prev_0 - 0.5 * (alpha_t * phi_1) * D1_0
            elif solver_type == "taylor":
                x_t = (
                    (sigma_t / sigma_prev_0) * x
                    - (alpha_t * phi_1) * model_prev_0
                    + (alpha_t * (phi_1 / h + 1.0)) * D1_0
                )
            else:
                raise ValueError("solver_type must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        else:
            phi_1 = torch.expm1(h)
            if solver_type == "dpmsolver":
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - 0.5 * (sigma_t * phi_1) * D1_0
                )
            elif solver_type == "taylor":
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - (sigma_t * (phi_1 / h - 1.0)) * D1_0
                )
            else:
                raise ValueError("solver_type must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        return x_t

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t):
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_2),
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1.0 / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            phi_2 = phi_1 / h + 1.0
            phi_3 = phi_2 / h - 0.5
            x_t = (
                (sigma_t / sigma_prev_0) * x
                - (alpha_t * phi_1) * model_prev_0
                + (alpha_t * phi_2) * D1
                - (alpha_t * phi_3) * D2
            )
        else:
            phi_1 = torch.expm1(h)
            phi_2 = phi_1 / h - 1.0
            phi_3 = phi_2 / h - 0.5
            x_t = (
                (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                - (sigma_t * phi_1) * model_prev_0
                - (sigma_t * phi_2) * D1
                - (sigma_t * phi_3) * D2
            )
        return x_t


class NoiseScheduleVP:
    def __init__(
        self,
        schedule="discrete",
        betas=None,
        alphas_cumprod=None,
        continuous_beta_0=0.1,
        continuous_beta_1=20.0,
        dtype=torch.float32,
    ):

        if schedule not in ["discrete", "linear"]:
            raise ValueError(
                "Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear'".format(schedule)
            )

        self.schedule = schedule
        if schedule == "discrete":
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.T = 1.0
            self.log_alpha_array = self.numerical_clip_alpha(log_alphas).reshape((1, -1)).to(dtype=dtype)
            self.total_N = self.log_alpha_array.shape[1]
            self.t_array = torch.linspace(0.0, 1.0, self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
        else:
            self.T = 1.0
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1

    def numerical_clip_alpha(self, log_alphas, clipped_lambda=-5.1):
        """
        For some beta schedules such as cosine schedule, the log-SNR has numerical isssues.
        We clip the log-SNR near t=T within -5.1 to ensure the stability.
        Such a trick is very useful for diffusion models with the cosine schedule, such as i-DDPM, guided-diffusion and GLIDE.
        """
        log_sigmas = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_alphas))
        lambs = log_alphas - log_sigmas
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == "discrete":
            return interpolate_fn(
                t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)
            ).reshape((-1))
        elif self.schedule == "linear":
            return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == "linear":
            tmp = 2.0 * (self.beta_1 - self.beta_0) * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == "discrete":
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2.0 * lamb)
            t = interpolate_fn(
                log_alpha.reshape((-1, 1)),
                torch.flip(self.log_alpha_array.to(lamb.device), [1]),
                torch.flip(self.t_array.to(lamb.device), [1]),
            )
            return t.reshape((-1,))
