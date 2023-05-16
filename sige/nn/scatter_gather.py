from typing import Dict, Optional

import torch

from .base import SIGEModule, SIGEModuleWrapper
from .gather import Gather
from .utils import activation


class ScatterGather(SIGEModule):
    def __init__(self, gather: Gather, activation_name: str = "identity", activation_first: bool = False):
        super(ScatterGather, self).__init__()
        self.gather = SIGEModuleWrapper(gather)
        self.activation_name = activation_name
        self.activation_first = activation_first

        self.load_runtime("scatter_gather")
        self.scatter_runtime = self.load_runtime("scatter", {})
        self.get_scatter_map_runtime = self.load_runtime("get_scatter_map", {})

        self.scatter_map = None
        self.output_res = None
        self.original_outputs = {}

    def forward(
        self, x: torch.Tensor, scale: Optional[torch.Tensor] = None, shift: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self.check_dtype(x, scale, shift)
        self.check_dim(x, scale, shift)
        b, c, h, w = x.shape
        active_indices = self.gather.module.active_indices
        block_size = self.gather.module.block_size
        if self.mode == "profile":
            output = torch.full(
                (self.original_outputs[self.cache_id].size(0) * active_indices.size(0), c, *block_size),
                fill_value=x[0, 0, 0, 0],
                dtype=x.dtype,
                device=x.device,
            )  # create a dummy gather output depending on the input for profiling
            if scale is not None:
                output = output * scale[0, 0, 0, 0]
            if shift is not None:
                output = output + shift[0, 0, 0, 0]
            output = activation(output, self.activation_name)
        elif self.mode == "full":
            output = x
            self.output_res = output.shape[2:]
            self.original_outputs[self.cache_id] = output.contiguous()
        elif self.mode == "sparse":
            device = x.device.type
            runtime = self.runtime[device]
            assert runtime is not None
            output = runtime(
                x.contiguous(),
                self.original_outputs[self.cache_id].contiguous(),
                block_size[0],
                block_size[1],
                active_indices.contiguous(),
                self.scatter_map.contiguous(),
                None if scale is None else scale.contiguous(),
                None if shift is None else shift.contiguous(),
                self.activation_name,
                self.activation_first,
            )
            if self.sparse_update:
                self.original_outputs[self.cache_id].copy_(
                    self.scatter_runtime[device](
                        x.contiguous(),
                        self.original_outputs[self.cache_id].contiguous(),
                        self.gather.module.offset[0],
                        self.gather.module.offset[1],
                        self.gather.module.model_stride[0],
                        self.gather.module.model_stride[1],
                        active_indices.contiguous(),
                        None,
                    )
                )
        else:
            raise NotImplementedError("Unknown mode: [%s]!!!" % self.mode)
        return output

    def clear_cache(self):
        self.original_outputs = {}

    def set_mask(self, masks: Dict, cache: Dict, timestamp: int):
        if self.timestamp != timestamp:
            super(ScatterGather, self).set_mask(masks, cache, timestamp)
            self.gather.module.set_mask(masks, cache, timestamp)

            mask = self.gather.module.mask
            h, w = mask.shape
            block_size = self.gather.module.block_size
            kernel_size = self.gather.module.kernel_size
            offset = self.gather.module.offset
            stride = self.gather.module.model_stride

            key = ("scatter_map", h, w, *block_size, *kernel_size, *offset, *stride)
            scatter_map = cache.get(key, None)
            if scatter_map is None:
                active_indices = self.gather.module.active_indices
                device = active_indices.device.type
                runtime = self.get_scatter_map_runtime[device]
                scatter_map = runtime(
                    h,
                    w,
                    block_size[0],
                    block_size[1],
                    kernel_size[0],
                    kernel_size[1],
                    offset[0],
                    offset[1],
                    stride[0],
                    stride[1],
                    active_indices,
                )
                cache[key] = scatter_map
            self.scatter_map = scatter_map
