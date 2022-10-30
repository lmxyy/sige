#include <torch/extension.h>

torch::Tensor scatter_cuda(
        const torch::Tensor &x,
        const torch::Tensor &y,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices, // Indices [N, 2], dim 0 is h, dim 1 is w,
        const torch::optional<torch::Tensor> &residual);

torch::Tensor scatter_with_block_residual_cuda(
        const torch::Tensor &x0, const torch::Tensor &y0,
        const torch::Tensor &x1, const torch::Tensor &y1,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices0,
        const torch::Tensor &activeIndices1);
