#include <iostream>
#include <torch/extension.h>
#include "MPSStream.h"
#include "MPSLibrary.h"
#include "MPSUtils.h"
#include "../common.cpp"

torch::Tensor scatter_gather_mps(
        const torch::Tensor &x, // [numActive, C, bSizeH1, bSizeH2]
        const torch::Tensor &y, // [B, C, H, W]
        int bSizeH, int bSizeW,
        const torch::Tensor &activeIndices, // [N, 2]
        const torch::Tensor &scatterMap,
        const torch::optional<torch::Tensor> &scale,
        const torch::optional<torch::Tensor> &shift,
        const std::string &activationName = std::string("identity"),
        bool activationFirst = false) {
    assert(x.size(1) == y.size(1));
    const int Ro = bSizeH, So = bSizeW;
    const int Rx = x.size(2), Sx = x.size(3);
    const int B = y.size(0), C = y.size(1), H = y.size(2), W = y.size(3);

    const int numActive = activeIndices.size(0);
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device()).requires_grad(false);
    auto output = torch::empty({B * numActive, C, Ro, So}, options);

    int scaleB = 0, scaleC = 0, scaleH = 0, scaleW = 0;
    if (scale.has_value()) {
        assert(broadcastable(y, scale.value()));
        scaleB = scale.value().size(0);
        scaleC = scale.value().size(1);
        scaleH = scale.value().size(2);
        scaleW = scale.value().size(3);
    }

    int shiftB = 0, shiftC = 0, shiftH = 0, shiftW = 0;
    if (shift.has_value()) {
        assert(broadcastable(y, shift.value()));
        shiftB = shift.value().size(0);
        shiftC = shift.value().size(1);
        shiftH = shift.value().size(2);
        shiftW = shift.value().size(3);
    }
    const auto activationType = getActivationType(activationName);

    const int total = output.numel();

    auto stream = at::mps::getCurrentMPSStream();
    auto library_manager = MPSLibraryManager::getInstance();
    char *library_path = getenv("SIGE_METAL_LIB_PATH");
    auto library = library_manager->getLibrary(library_path);
    auto func_pso = library->getComputePipelineState("scatter_gather_mps_kernel");

    // create command buffer and encoder
    MTLCommandBuffer_t command_buffer = stream->commandBuffer();
    MTLComputeCommandEncoder_t compute_encoder = [command_buffer computeCommandEncoder];

    setMTLArgs(compute_encoder, func_pso,
               total, numActive,
               B, C, H, W,
               Rx, Sx,
               Ro, So,
               x, y, output,
               activeIndices, scatterMap,
               scale.has_value() ? scale.value() : x,
               scaleB, scaleC, scaleH, scaleW,
               shift.has_value() ? shift.value() : x,
               shiftB, shiftC, shiftH, shiftW,
               activationType, activationFirst);
    MTLSize grid_size = MTLSizeMake(total, 1, 1);
    NSUInteger thread_group_size_x = func_pso.maxTotalThreadsPerThreadgroup;
    if (thread_group_size_x > total)
        thread_group_size_x = total;
    MTLSize thread_group_size = MTLSizeMake(thread_group_size_x, 1, 1);
    [compute_encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
    [compute_encoder endEncoding];
    stream->commit(true);

    return output;
}

torch::Tensor get_scatter_map_mps(
        int H, int W,
        int bSizeH, int bSizeW,
        int kSizeH, int kSizeW,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices) {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(activeIndices.device()).requires_grad(false);
    auto scatterMap = torch::full({H, W, 3}, -1, options);
    const int R = (bSizeH - kSizeH) / strideH + 1, S = (bSizeW - kSizeW) / strideW + 1;
    const int numActive = activeIndices.size(0);
    const int total = numActive * R * S;

        auto stream = at::mps::getCurrentMPSStream();
    auto library_manager = MPSLibraryManager::getInstance();
    char *library_path = getenv("SIGE_METAL_LIB_PATH");
    auto library = library_manager->getLibrary(library_path);
    auto func_pso = library->getComputePipelineState("get_scatter_map_mps_kernel");

    // create command buffer and encoder
    MTLCommandBuffer_t command_buffer = stream->commandBuffer();
    MTLComputeCommandEncoder_t compute_encoder = [command_buffer computeCommandEncoder];

    setMTLArgs(compute_encoder, func_pso,
               total,
               H, W,
               R, S,
               offsetH, offsetW,
               strideH, strideW,
               scatterMap,
               activeIndices);
    MTLSize grid_size = MTLSizeMake(total, 1, 1);
    NSUInteger thread_group_size_x = func_pso.maxTotalThreadsPerThreadgroup;
    if (thread_group_size_x > total)
        thread_group_size_x = total;
    MTLSize thread_group_size = MTLSizeMake(thread_group_size_x, 1, 1);
    [compute_encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
    [compute_encoder endEncoding];
    stream->commit(true);

    return scatterMap;
}