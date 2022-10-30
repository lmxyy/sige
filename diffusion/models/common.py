import math

import torch
from torch import nn
from torch.nn import functional as F


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def swish(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def my_group_norm(x: torch.Tensor, norm: nn.GroupNorm):
    n, c, h, w = x.shape
    assert n == 1
    num_groups = norm.num_groups
    group_size = c // num_groups
    x = x.view([n, num_groups, group_size, h, w])
    var, mean = torch.var_mean(x, unbiased=False, dim=[2, 3, 4], keepdim=True)
    var = torch.sqrt(var + norm.eps)
    x = (x - mean) / var
    scale = 1 / var[0, :, 0, 0]
    shift = -(mean / var)[0, :, 0, 0]
    scale = scale.repeat_interleave(group_size)
    shift = shift.repeat_interleave(group_size)
    if norm.affine:
        x = x.view(n, c, h, w)
        x = x * norm.weight.view(1, -1, 1, 1)
        x = x + norm.bias.view(1, -1, 1, 1)
        scale = scale * norm.weight
        shift = shift * norm.weight
        shift = shift + norm.bias
    return x, scale, shift
