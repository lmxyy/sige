#include "common_cpu.cpp"
#include <torch/extension.h>

void scatter_kernel(
        int numActive,
        int B, int C, int H, int W,
        int R, int S,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const float *x, float *output,
        const int *activeIndices,
        const float *residual,
        int residualB, int residualC, int residualH, int residualW) {
#pragma omp parallel for collapse(3)
    for (int bb = 0; bb < B; ++bb)
        for (int ib = 0; ib < numActive; ++ib)
            for (int cc = 0; cc < C; ++cc) {
                int biH = (offsetH + activeIndices[ib << 1]) / strideH;
                int biW = (offsetW + activeIndices[ib << 1 | 1]) / strideW;
                for (int intraBh = 0; intraBh < R; ++intraBh) {
                    int hh = biH + intraBh;
                    if (hh >= H)
                        break;
                    for (int intraBw = 0; intraBw < S; ++intraBw) {
                        int ww = biW + intraBw;
                        if (ww >= W)
                            break;
                        int index = (bb * numActive + ib) * C * R * S + cc * R * S + intraBh * S + intraBw;
                        auto p = bb * C * H * W + cc * H * W + hh * W + ww;
                        auto z = x[index];
                        z = binary_op_array<ADD>(
                                residual, z,
                                residualB, residualC, residualH, residualW,
                                bb, cc, hh, ww);
                        output[p] = z;
                    }
                }
            }
}

void calibrate_residual_kernel(
        int numActive,
        int B, int C, int H, int W,
        int R, int S,
        const float *x, const float *y,
        float *output,
        const int *activeIndices) {
#pragma omp parallel for collapse(3)
    for (int bb = 0; bb < B; ++bb)
        for (int ib = 0; ib < numActive; ++ib)
            for (int cc = 0; cc < C; ++cc) {
                int biH = activeIndices[ib << 1];
                int biW = activeIndices[ib << 1 | 1];
                for (int intraBh = 0; intraBh < R; ++intraBh) {
                    int hh = biH + intraBh;
                    if (hh >= H)
                        break;
                    for (int intraBw = 0; intraBw < S; ++intraBw) {
                        int ww = biW + intraBw;
                        if (ww >= W)
                            break;
                        int index = (bb * numActive + ib) * C * R * S + cc * R * S + intraBh * S + intraBw;
                        int p = bb * C * H * W + cc * H * W + hh * W + ww;
                        output[p] += x[index] - y[p];
                    }
                }
            }
}

torch::Tensor scatter_cpu(
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

    scatter_kernel(
            numActive,
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

torch::Tensor scatter_with_block_residual_cpu(
        const torch::Tensor &x0, const torch::Tensor &y0,
        const torch::Tensor &x1, const torch::Tensor &y1,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices0,
        const torch::Tensor &activeIndices1) {
    auto output = scatter_cpu(x0, y0, offsetH, offsetW, strideH, strideW, activeIndices0, y1);
    auto outputData = output.data_ptr<float>();
    auto x1Data = x1.data_ptr<float>(), y1Data = y1.data_ptr<float>();
    auto activeIndicesData = activeIndices1.data_ptr<int>();
    const int C = x1.size(1), R = x1.size(2), S = x1.size(3);
    const int B = y1.size(0), H = y1.size(2), W = y1.size(3);
    const int numActive = activeIndices1.size(0);

    calibrate_residual_kernel(
            numActive,
            B, C, H, W,
            R, S,
            x1Data, y1Data,
            outputData,
            activeIndicesData);

    return output;
}
