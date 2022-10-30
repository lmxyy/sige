import warnings
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn

from sige.utils import reduce_mask
from .base import SIGEModule
from .utils import activation


class Gather(SIGEModule):
    def __init__(
        self,
        conv: nn.Conv2d,
        block_size: Union[int, Tuple[int, int]],
        offset: Optional[Union[int, Tuple[int, int]]] = None,
        activation_name: str = "identity",
        activation_first: bool = False,
        verbose: bool = False,
    ):
        super(Gather, self).__init__()
        if isinstance(block_size, int):
            block_size = (block_size, block_size)

        n0 = max(block_size[0] - conv.kernel_size[0], 0) // conv.stride[0]
        n1 = max(block_size[1] - conv.kernel_size[1], 0) // conv.stride[1]
        b0 = n0 * conv.stride[0] + conv.kernel_size[0]
        b1 = n1 * conv.stride[1] + conv.kernel_size[1]
        if (b0, b1) != block_size:
            warnings.warn("Change the block size from (%d, %d) to (%d, %d)" % (*block_size, b0, b1))

        self.model_stride = conv.stride
        self.kernel_size = conv.kernel_size

        self.block_size = (b0, b1)
        self.block_stride = ((n0 + 1) * conv.stride[0], (n1 + 1) * conv.stride[1])
        if offset is None:
            self.offset = conv.padding
        else:
            if isinstance(offset, int):
                offset = (offset, offset)
            self.offset = offset
        self.activation_name = activation_name
        self.activation_first = activation_first
        self.verbose = verbose

        self.load_runtime("gather")

        self.input_res: Optional[Tuple[int, int]] = None
        self.active_indices: Optional[torch.Tensor] = None

    def forward(
        self, x: torch.Tensor, scale: Optional[torch.Tensor] = None, shift: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self.check_dtype(x, scale, shift)
        self.check_dim(x, scale, shift)
        b, c, h, w = x.shape
        if self.mode == "profile":
            output = torch.full(
                (b * self.active_indices.size(0), c, *self.block_size),
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
            self.input_res = x.shape[2:]
            assert scale is None
            assert shift is None
            output = x
        elif self.mode == "sparse":
            device = x.device.type
            runtime = self.runtime[device]
            assert runtime is not None
            output = runtime(
                x.contiguous(),
                self.block_size[0],
                self.block_size[1],
                self.active_indices.contiguous(),
                None if scale is None else scale.contiguous(),
                None if shift is None else shift.contiguous(),
                self.activation_name,
                self.activation_first,
            )
        else:
            raise NotImplementedError("Unknown mode: [%s]!!!" % self.mode)
        return output

    def set_mask(self, masks: Dict, cache: Dict, timestamp: int):
        if self.timestamp != timestamp:
            super(Gather, self).set_mask(masks, cache, timestamp)
            assert self.input_res is not None
            res = tuple(self.input_res)
            mask = masks[res]
            self.mask = mask
            key = ("active_indices", *res, *self.block_size, *self.block_stride, *self.offset)
            active_indices = cache.get(key, None)
            if active_indices is None:
                active_indices = reduce_mask(
                    mask, self.block_size, self.block_stride, self.offset, verbose=self.verbose
                )
                cache[key] = active_indices
            self.active_indices = active_indices
