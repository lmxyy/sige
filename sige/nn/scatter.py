from typing import Optional

import torch

from .base import SIGEModule, SIGEModuleWrapper
from .gather import Gather


class Scatter(SIGEModule):
    def __init__(self, gather: Gather):
        super(Scatter, self).__init__()
        self.gather = SIGEModuleWrapper(gather)

        self.load_runtime("scatter")
        self.output_res = None
        self.original_output = None

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.check_dtype(x, residual)
        self.check_dim(x, residual)
        if self.mode == "profile":
            _, c, _, _ = x.shape
            output = torch.full(
                (self.original_output.size(0), c, *self.output_res),
                fill_value=x[0, 0, 0, 0],
                dtype=x.dtype,
                device=x.device,
            )  # create a dummy scatter output depending on the input for profiling
            if residual is not None:
                output = output + residual
        elif self.mode == "full":
            if residual is None:
                output = x
            else:
                output = x + residual
            self.output_res = output.shape[2:]
            self.original_output = output.contiguous()
        elif self.mode == "sparse":
            device = x.device.type
            runtime = self.runtime[device]
            assert runtime is not None

            active_indices = self.gather.module.active_indices
            offset = self.gather.module.offset
            stride = self.gather.module.model_stride
            output = runtime(
                x.contiguous(),
                self.original_output.contiguous(),
                offset[0],
                offset[1],
                stride[0],
                stride[1],
                active_indices.contiguous(),
                None if residual is None else residual.contiguous(),
            )
        else:
            raise NotImplementedError("Unknown mode: [%s]!!!" % self.mode)
        return output


class ScatterWithBlockResidual(SIGEModule):
    def __init__(self, main_gather: Gather, shortcut_gather: Gather):
        super(ScatterWithBlockResidual, self).__init__()
        self.main_gather = SIGEModuleWrapper(main_gather)
        self.shortcut_gather = SIGEModuleWrapper(shortcut_gather)

        self.load_runtime("scatter_with_block_residual")
        self.output_res = None
        self.original_output = None
        self.original_residual = None

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        self.check_dtype(x, residual)
        self.check_dim(x, residual)
        if self.mode == "profile":
            _, c, _, _ = x.shape
            output = torch.full(
                (self.original_output.size(0), c, *self.output_res),
                fill_value=x[0, 0, 0, 0] + residual[0, 0, 0, 0],
                dtype=x.dtype,
                device=x.device,
            )
        elif self.mode == "full":
            output = x + residual
            self.output_res = output.shape[2:]
            self.original_output = output.contiguous()
            self.original_residual = residual.contiguous()
        elif self.mode == "sparse":
            device = x.device.type
            runtime = self.runtime[device]
            assert runtime is not None

            offset = self.main_gather.module.offset
            stride = self.main_gather.module.model_stride

            output = runtime(
                x.contiguous(),
                self.original_output.contiguous(),
                residual.contiguous(),
                self.original_residual.contiguous(),
                offset[0],
                offset[1],
                stride[0],
                stride[1],
                self.main_gather.module.active_indices.contiguous(),
                self.shortcut_gather.module.active_indices.contiguous(),
            )
        return output
