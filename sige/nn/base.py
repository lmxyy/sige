import importlib
from typing import Dict, List, Optional
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class SIGEModule(nn.Module):
    def __init__(self, call_super: bool = True):
        if call_super:
            super(SIGEModule, self).__init__()
        self.devices: List[str] = ["cpu", "cuda"]
        self.supported_dtypes = [torch.float32]
        self.mode: str = "full"
        self.runtime: Dict = {}
        self.mask: Optional[torch.Tensor] = None
        self.timestamp = None

    def set_mask(self, masks: Dict, cache: Dict, timestamp: int):
        self.timestamp = timestamp

    def load_runtime(self, function_name: str, runtime_dict: Dict = None):
        if runtime_dict is None:
            runtime_dict = self.runtime
        for device in self.devices:
            name = "sige.%s" % device
            try:
                module = importlib.import_module(name)
                runtime = getattr(module, function_name)
                runtime_dict[device] = runtime
            except (ModuleNotFoundError, AttributeError):
                runtime_dict[device] = None
        return runtime_dict

    def set_mode(self, mode: str):
        self.mode = mode

    def check_dtype(self, *args):
        for x in args:
            if x is not None:
                assert isinstance(x, torch.Tensor)
                if x.dtype not in self.supported_dtypes:
                    raise NotImplementedError(
                        "[%s] does not support dtype [%s]!!! "
                        "Currently supported dtype %s." % (self.__class__.__name__, x.dtype, str(self.supported_dtypes))
                    )

    def check_dim(self, *args):
        for x in args:
            if x is not None:
                assert isinstance(x, torch.Tensor)
                if x.dim() != 4:
                    raise NotImplementedError(
                        "[%s] does not support input with dim [%d]!!!" % (self.__class__.__name__, x.dim())
                    )


class SIGEModuleWrapper:
    def __init__(self, module: SIGEModule):
        self.module = module


class SIGEConv2d(nn.Conv2d, SIGEModule):
    def __init__(self, *args, **kwargs):
        nn.Conv2d.__init__(self, *args, **kwargs)
        SIGEModule.__init__(self, call_super=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "full":
            output = super(SIGEConv2d, self).forward(x)
        elif self.mode in ["sparse", "profile"]:
            output = F.conv2d(x, self.weight, self.bias, self.stride, (0, 0), self.dilation, self.groups)
        else:
            raise NotImplementedError("Unknown mode: %s" % self.mode)
        return output


class SIGEModel(nn.Module):
    def __init__(self, call_super: bool = True):
        if call_super:
            super(SIGEModel, self).__init__()
        self.mode = "full"
        self.timestamp = 0

    def set_masks(self, masks: Dict[Tuple[int, int], torch.Tensor]):
        self.timestamp += 1
        timestamp = self.timestamp
        cache = {}
        for module in self.modules():
            if isinstance(module, SIGEModule):
                module.set_mask(masks, cache, timestamp)

    def set_mode(self, mode: str):
        self.mode = mode
        for module in self.modules():
            if isinstance(module, SIGEModule):
                module.set_mode(mode)
