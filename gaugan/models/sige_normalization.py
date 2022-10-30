import re
from typing import Optional

import torch
from torch import nn

from sige.nn import Gather, SIGEConv2d, SIGEModule, Scatter, ScatterGather
from .mobile_modules import SIGESeparableConv2d
from .sync_batchnorm import SynchronizedBatchNorm2d


class SIGEFusedSPADE(SIGEModule):
    def __init__(
        self,
        config_text,
        norm_nc,
        nhidden: int = 128,
        seg_gather: Optional[Gather] = None,
        shortcut_conv: Optional[nn.Conv2d] = None,
        main_block_size: Optional[int] = 6,
        shortcut_block_size: Optional[int] = 4,
    ):
        super(SIGEFusedSPADE, self).__init__()
        is_shortcut = shortcut_conv is not None

        self.norm_nc = norm_nc
        self.is_shortcut = is_shortcut

        assert config_text.startswith("spade")
        parsed = re.search(r"spade(\D+)(\d)x\d", config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        assert ks == 3

        if param_free_norm_type == "instance":
            raise NotImplementedError
        elif param_free_norm_type == "syncbatch":
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == "batch":
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError("%s is not a recognized param-free norm type in SPADE" % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.

        pw = ks // 2

        block_size = shortcut_block_size if is_shortcut else main_block_size
        self.support_sparse = seg_gather is not None
        Conv2d = SIGEConv2d if self.support_sparse else nn.Conv2d
        self.mlp_gamma_beta = Conv2d(nhidden, 2 * norm_nc, kernel_size=ks, padding=pw)

        if self.support_sparse:
            if is_shortcut:
                self.scatter = Scatter(seg_gather)
                self.gather = Gather(shortcut_conv, block_size)
            else:
                self.scatter_gather = ScatterGather(seg_gather)
        self.scale, self.shift = None, None

    def forward(self, x, actv):
        if self.mode == "full":
            normalized = self.param_free_norm(x)
            mean = self.param_free_norm.running_mean
            var = self.param_free_norm.running_var
            var = torch.sqrt(var + self.param_free_norm.eps)
            self.scale = 1 / var
            self.shift = -(mean / var)
        elif self.mode in ("sparse", "profile"):
            normalized = x
        else:
            raise NotImplementedError("Unknown mode [%s]!!!" % self.mode)

        # Part 2. produce scaling and bias conditioned on semantic map
        gamma_beta = self.mlp_gamma_beta(actv)

        if self.support_sparse:
            if self.is_shortcut:
                gamma_beta = self.scatter(gamma_beta)
                gamma_beta = self.gather(gamma_beta)
            else:
                gamma_beta = self.scatter_gather(gamma_beta)

        gamma, beta = torch.split(gamma_beta, self.norm_nc, dim=1)
        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SIGEFusedSubMobileSPADE(SIGEModule):
    def __init__(
        self,
        config_text,
        norm_nc,
        nhidden,
        oc,
        seg_gather: Optional[Gather] = None,
        shortcut_conv: Optional[nn.Conv2d] = None,
        main_block_size: Optional[int] = 6,
        shortcut_block_size: Optional[int] = 4,
    ):
        super(SIGEFusedSubMobileSPADE, self).__init__()
        is_shortcut = shortcut_conv is not None
        self.is_shortcut = is_shortcut

        assert config_text.startswith("spade")
        parsed = re.search(r"spade(\D+)(\d)x\d", config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        assert ks == 3

        if param_free_norm_type == "syncbatch":
            assert norm_nc >= oc
            self.param_free_norm = SynchronizedBatchNorm2d(oc, affine=False)
        else:
            raise ValueError("%s is not a recognized param-free norm type in SPADE" % param_free_norm_type)

        block_size = shortcut_block_size if is_shortcut else main_block_size
        self.support_sparse = seg_gather is not None

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        pw = ks // 2
        self.mlp_gamma = SIGESeparableConv2d(
            nhidden, oc, kernel_size=ks, padding=pw, support_sparse=self.support_sparse
        )
        self.mlp_beta = SIGESeparableConv2d(nhidden, oc, kernel_size=ks, padding=pw, support_sparse=self.support_sparse)

        if self.support_sparse:
            if is_shortcut:
                self.scatter_gamma = Scatter(seg_gather)
                self.gather_gamma = Gather(shortcut_conv, block_size)
                self.scatter_beta = Scatter(seg_gather)
                self.gather_beta = Gather(shortcut_conv, block_size)
            else:
                self.scatter_gather_gamma = ScatterGather(seg_gather)
                self.scatter_gather_beta = ScatterGather(seg_gather)
        self.scale, self.shift = None, None

    def forward(self, x, actv):
        if self.mode == "full":
            normalized = self.param_free_norm(x)
            mean = self.param_free_norm.running_mean
            var = self.param_free_norm.running_var
            var = torch.sqrt(var + self.param_free_norm.eps)
            self.scale = 1 / var
            self.shift = -(mean / var)
        elif self.mode in ("sparse", "profile"):
            normalized = x
        else:
            raise NotImplementedError("Unknown mode [%s]!!!" % self.mode)

        # Part 2. produce scaling and bias conditioned on semantic map
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        if self.support_sparse:
            if self.is_shortcut:
                gamma = self.scatter_gamma(gamma)
                gamma = self.gather_gamma(gamma)
                beta = self.scatter_beta(beta)
                beta = self.gather_beta(beta)
            else:
                gamma = self.scatter_gather_gamma(gamma)
                beta = self.scatter_gather_beta(beta)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
