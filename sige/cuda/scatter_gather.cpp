#include <torch/extension.h>

torch::Tensor scatter_gather_cuda(
        const torch::Tensor &x, // [B, C, bSizeH1, bSizeH2]
        const torch::Tensor &y, // [1, C, H, W]
        int bSizeH, int bSizeW,
        const torch::Tensor &activeIndices, // [N, 2]
        const torch::Tensor &scatterMap,
        const torch::optional<torch::Tensor> &scale,
        const torch::optional<torch::Tensor> &shift,
        const std::string &activationName = std::string("identity"),
        bool activationFirst = false);

torch::Tensor get_scatter_map_cuda(
        int H, int W,
        int bSizeH, int bSizeW,
        int kSizeH, int kSizeW,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices);