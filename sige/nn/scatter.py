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
        self.original_outputs = {}

    def clear_cache(self):
        self.original_outputs = {}

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.check_dtype(x, residual)
        self.check_dim(x, residual)
        if self.mode == "profile":
            _, c, _, _ = x.shape
            output = torch.full(
                (self.original_outputs[self.cache_id].size(0), c, *self.output_res),
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
            self.original_outputs[self.cache_id] = output.contiguous()
        elif self.mode == "sparse":
            device = x.device.type
            runtime = self.runtime[device]
            assert runtime is not None

            active_indices = self.gather.module.active_indices
            offset = self.gather.module.offset
            stride = self.gather.module.model_stride
            output = runtime(
                x.contiguous(),
                self.original_outputs[self.cache_id].contiguous(),
                offset[0],
                offset[1],
                stride[0],
                stride[1],
                active_indices.contiguous(),
                None if residual is None else residual.contiguous(),
            )
            if self.sparse_update:
                self.original_outputs[self.cache_id].copy_(output.contiguous())
        else:
            raise NotImplementedError("Unknown mode: [%s]!!!" % self.mode)
        return output


class ScatterWithBlockResidual(SIGEModule):
    def __init__(self, main_gather: Gather, shortcut_gather: Gather):
        super(ScatterWithBlockResidual, self).__init__()
        self.main_gather = SIGEModuleWrapper(main_gather)
        self.shortcut_gather = SIGEModuleWrapper(shortcut_gather)

        self.load_runtime("scatter_with_block_residual")
        self.scatter_runtime = None
        self.output_res = None
        self.original_outputs = {}
        self.original_residuals = {}

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        self.check_dtype(x, residual)
        self.check_dim(x, residual)
        if self.mode == "profile":
            _, c, _, _ = x.shape
            output = torch.full(
                (self.original_outputs[self.cache_id].size(0), c, *self.output_res),
                fill_value=x[0, 0, 0, 0] + residual[0, 0, 0, 0],
                dtype=x.dtype,
                device=x.device,
            )
        elif self.mode == "full":
            output = x + residual
            self.output_res = output.shape[2:]
            self.original_outputs[self.cache_id] = output.contiguous()
            self.original_residuals[self.cache_id] = residual.contiguous()
        elif self.mode == "sparse":
            device = x.device.type
            runtime = self.runtime[device]
            assert runtime is not None

            offset = self.main_gather.module.offset
            stride = self.main_gather.module.model_stride

            output = runtime(
                x.contiguous(),
                self.original_outputs[self.cache_id].contiguous(),
                residual.contiguous(),
                self.original_residuals[self.cache_id].contiguous(),
                offset[0],
                offset[1],
                stride[0],
                stride[1],
                self.main_gather.module.active_indices.contiguous(),
                self.shortcut_gather.module.active_indices.contiguous(),
            )
            if self.sparse_update:
                if self.scatter_runtime is None:
                    self.scatter_runtime = self.load_runtime("scatter", {})
                self.original_outputs[self.cache_id].copy_(output.contiguous())
                self.original_residuals[self.cache_id].copy_(
                    self.scatter_runtime[device](
                        residual.contiguous(),
                        self.original_residuals[self.cache_id].contiguous(),
                        self.shortcut_gather.module.offset[0],
                        self.shortcut_gather.module.offset[1],
                        self.shortcut_gather.module.model_stride[0],
                        self.shortcut_gather.module.model_stride[1],
                        self.shortcut_gather.module.active_indices.contiguous(),
                        None,
                    )
                )
        else:
            raise NotImplementedError("Unknown mode: [%s]!!!" % self.mode)
        return output

    def clear_cache(self):
        self.original_outputs = {}
        self.original_residuals = {}
