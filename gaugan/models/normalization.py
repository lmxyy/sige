"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import functools
import re

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from .mobile_modules import SeparableConv2d
from .sync_batchnorm import SynchronizedBatchNorm2d


def get_norm_layer(norm_type="instance"):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":
        return lambda x: x
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


# Returns a function that creates a normalization function
# that does not condition on semantic map


def get_nonspade_norm_layer(opt, norm_type="instance"):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, "out_channels"):
            return getattr(layer, "out_channels")
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith("spectral"):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len("spectral") :]

        if subnorm_type == "none" or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, "bias", None) is not None:
            delattr(layer, "bias")
            layer.register_parameter("bias", None)

        if subnorm_type == "batch":
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == "sync_batch":
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == "instance":
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError("normalization layer %s is not recognized" % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, nhidden=128):
        super(SPADE, self).__init__()

        assert config_text.startswith("spade")
        parsed = re.search(r"spade(\D+)(\d)x\d", config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == "instance":
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == "syncbatch":
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == "batch":
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError("%s is not a recognized param-free norm type in SPADE" % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.

        pw = ks // 2
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class FusedSPADE(nn.Module):
    def __init__(self, config_text, norm_nc, nhidden=128):
        super(FusedSPADE, self).__init__()
        self.norm_nc = norm_nc

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
        self.mlp_gamma_beta = nn.Conv2d(nhidden, 2 * norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, actv):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        gamma_beta = self.mlp_gamma_beta(actv)
        gamma, beta = torch.split(gamma_beta, self.norm_nc, dim=1)
        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class MobileSPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, nhidden=128, separable_conv_norm="none"):
        super(MobileSPADE, self).__init__()

        assert config_text.startswith("spade")
        parsed = re.search(r"spade(\D+)(\d)x\d", config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == "instance":
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == "syncbatch":
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == "batch":
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError("%s is not a recognized param-free norm type in SPADE" % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.

        pw = ks // 2
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        norm_layer = get_norm_layer(separable_conv_norm)
        self.mlp_gamma = SeparableConv2d(nhidden, norm_nc, kernel_size=ks, padding=pw, norm_layer=norm_layer)
        self.mlp_beta = SeparableConv2d(nhidden, norm_nc, kernel_size=ks, padding=pw, norm_layer=norm_layer)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SubMobileSPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, nhidden, oc):
        super(SubMobileSPADE, self).__init__()

        assert config_text.startswith("spade")
        parsed = re.search(r"spade(\D+)(\d)x\d", config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == "syncbatch":
            assert norm_nc >= oc
            self.param_free_norm = SynchronizedBatchNorm2d(oc, affine=False)
        else:
            raise ValueError("%s is not a recognized param-free norm type in SPADE" % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.

        pw = ks // 2
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma = SeparableConv2d(nhidden, oc, kernel_size=ks, padding=pw)
        self.mlp_beta = SeparableConv2d(nhidden, oc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class FusedSubMobileSPADE(nn.Module):
    def __init__(self, config_text, norm_nc, nhidden, oc):
        super(FusedSubMobileSPADE, self).__init__()

        assert config_text.startswith("spade")
        parsed = re.search(r"spade(\D+)(\d)x\d", config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == "syncbatch":
            assert norm_nc >= oc
            self.param_free_norm = SynchronizedBatchNorm2d(oc, affine=False)
        else:
            raise ValueError("%s is not a recognized param-free norm type in SPADE" % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.

        pw = ks // 2
        self.mlp_gamma = SeparableConv2d(nhidden, oc, kernel_size=ks, padding=pw)
        self.mlp_beta = SeparableConv2d(nhidden, oc, kernel_size=ks, padding=pw)

    def forward(self, x, actv):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out
