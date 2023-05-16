#include <iostream>
#include <torch/extension.h>
#include "MPSStream.h"
#include "MPSLibrary.h"
#include "MPSUtils.h"
#include "../common.cpp"

torch::Tensor scatter_mps(
        const torch::Tensor &x,
        const torch::Tensor &y,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices, // Indices [N, 2], dim 0 is h, dim 1 is w,
        const torch::optional<torch::Tensor> &residual) {
    const int numActive = activeIndices.size(0);

    const int C = x.size(1), R = x.size(2), S = x.size(3);
    const int B = y.size(0), H = y.size(2), W = y.size(3);
    auto output = y.clone();

    int residualB = 0, residualC = 0, residualH = 0, residualW = 0;
    if (residual.has_value()) {
        assert(broadcastable(y, residual.value()));
        residualB = residual.value().size(0);
        residualC = residual.value().size(1);
        residualH = residual.value().size(2);
        residualW = residual.value().size(3);
    }

    const int total = x.numel();

    auto stream = at::mps::getCurrentMPSStream();
    auto library_manager = MPSLibraryManager::getInstance();
    char *library_path = getenv("SIGE_METAL_LIB_PATH");
    auto library = library_manager->getLibrary(library_path);
    auto func_pso = library->getComputePipelineState("scatter_mps_kernel");

    // create command buffer and encoder
    MTLCommandBuffer_t command_buffer = stream->commandBuffer();
    MTLComputeCommandEncoder_t compute_encoder = [command_buffer computeCommandEncoder];

    setMTLArgs(compute_encoder, func_pso,
               total, numActive,
               B, C, H, W,
               R, S,
               offsetH, offsetW,
               strideH, strideW,
               x, output,
               activeIndices,
               residual.has_value() ? residual.value():x,
               residualB, residualC, residualH, residualW);
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

torch::Tensor scatter_with_block_residual_mps(
        const torch::Tensor &x0, const torch::Tensor &y0,
        const torch::Tensor &x1, const torch::Tensor &y1,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices0,
        const torch::Tensor &activeIndices1) {
    auto output = scatter_mps(x0, y0, offsetH, offsetW, strideH, strideW, activeIndices0, y1);
    const int C = x1.size(1), R = x1.size(2), S = x1.size(3);
    const int B = y1.size(0), H = y1.size(2), W = y1.size(3);
    const int numActive = activeIndices1.size(0);

    const int total = x1.numel();

    auto stream = at::mps::getCurrentMPSStream();
    auto library_manager = MPSLibraryManager::getInstance();
    char *library_path = getenv("SIGE_METAL_LIB_PATH");
    auto library = library_manager->getLibrary(library_path);
    auto func_pso = library->getComputePipelineState("calibrate_residual_mps_kernel");

    // create command buffer and encoder
    MTLCommandBuffer_t command_buffer = stream->commandBuffer();
    MTLComputeCommandEncoder_t compute_encoder = [command_buffer computeCommandEncoder];

    setMTLArgs(compute_encoder, func_pso,
               total, numActive,
               B, C, H, W,
               R, S,
               x1, y1,
               output,
               activeIndices1);
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