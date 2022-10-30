#include "common_cuda.cu"
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>

__global__ void scatter_cuda_kernel(
        int total, int numActive,
        int B, int C, int H, int W,
        int R, int S,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const float *__restrict__ x, float *__restrict__ output,
        const int *__restrict__ activeIndices,
        const float *__restrict__ residual,
        int residualB, int residualC, int residualH, int residualW) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total)
        return;
    int t = index;
    int intraBw = t % S;
    t /= S;
    int intraBh = t % R;
    t /= R;
    int cc = t % C;
    t /= C;
    int ib = t % numActive, bb = t / numActive;
    int biH = (offsetH + activeIndices[ib << 1]) / strideH;
    int hh = biH + intraBh;
    if (hh >= H)
        return;
    int biW = (offsetW + activeIndices[ib << 1 | 1]) / strideW;
    int ww = biW + intraBw;
    if (ww >= W)
        return;
    auto p = bb * C * H * W + cc * H * W + hh * W + ww;
    auto z = x[index];
    z = binary_op_array_cuda<ADD>(
            residual, z,
            residualB, residualC, residualH, residualW,
            bb, cc, hh, ww);
    output[p] = z;
}

__global__ void calibrate_residual_cuda_kernel(
        int total, int numActive,
        int B, int C, int H, int W,
        int R, int S,
        const float *__restrict__ x, const float *__restrict__ y,
        float *__restrict__ output,
        const int *__restrict__ activeIndices) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total)
        return;
    int t = index;
    int intraBw = t % S;
    t /= S;
    int intraBh = t % R;
    t /= R;
    int cc = t % C;
    t /= C;
    int ib = t % numActive, bb = t / numActive;
    int biH = activeIndices[ib << 1];
    int hh = biH + intraBh;
    if (hh >= H)
        return;
    int biW = activeIndices[ib << 1 | 1];
    int ww = biW + intraBw;
    if (ww >= W)
        return;
    int p = bb * C * H * W + cc * H * W + hh * W + ww;
    output[p] += x[index] - y[p];
}

torch::Tensor scatter_cuda(
        const torch::Tensor &x,
        const torch::Tensor &y,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices, // Indices [N, 2], dim 0 is h, dim 1 is w,
        const torch::optional<torch::Tensor> &residual) {
    const int numActive = activeIndices.size(0);
    auto xData = x.data_ptr<float>();
    auto activeIndicesData = activeIndices.data_ptr<int>();

    const int C = x.size(1), R = x.size(2), S = x.size(3);
    const int B = y.size(0), H = y.size(2), W = y.size(3);
    auto output = y.clone();
    auto outputData = output.data_ptr<float>();

    const float *residualData = nullptr;
    int residualB = 0, residualC = 0, residualH = 0, residualW = 0;
    if (residual.has_value()) {
        assert(broadcastable(y, residual.value()));
        residualData = residual.value().data_ptr<float>();
        residualB = residual.value().size(0);
        residualC = residual.value().size(1);
        residualH = residual.value().size(2);
        residualW = residual.value().size(3);
    }

    const int total = x.numel();
    const dim3 blocks((total + threads - 1) / threads, 1);
    scatter_cuda_kernel<<<blocks, threads>>>(
            total, numActive,
            B, C, H, W,
            R, S,
            offsetH, offsetW,
            strideH, strideW,
            xData, outputData,
            activeIndicesData,
            residualData,
            residualB, residualC, residualH, residualW);

    return output;
}

torch::Tensor scatter_with_block_residual_cuda(
        const torch::Tensor &x0, const torch::Tensor &y0,
        const torch::Tensor &x1, const torch::Tensor &y1,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices0,
        const torch::Tensor &activeIndices1) {
    auto output = scatter_cuda(x0, y0, offsetH, offsetW, strideH, strideW, activeIndices0, y1);
    auto outputData = output.data_ptr<float>();
    auto x1Data = x1.data_ptr<float>(), y1Data = y1.data_ptr<float>();
    auto activeIndicesData = activeIndices1.data_ptr<int>();
    const int C = x1.size(1), R = x1.size(2), S = x1.size(3);
    const int B = y1.size(0), H = y1.size(2), W = y1.size(3);
    const int numActive = activeIndices1.size(0);

    const int total = x1.numel();
    const dim3 blocks((total + threads - 1) / threads, 1);

    calibrate_residual_cuda_kernel<<<blocks, threads>>>(
            total, numActive,
            B, C, H, W,
            R, S,
            x1Data, y1Data,
            outputData,
            activeIndicesData);

    return output;
}
