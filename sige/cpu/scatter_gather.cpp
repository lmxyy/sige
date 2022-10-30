#include "common_cpu.cpp"
#include <iostream>
#include <torch/extension.h>

void scatter_gather_cpu_kernel(
        int numActive,
        int B, int C, int H, int W,
        int Rx, int Sx, // The height and width of x
        int Ro, int So, // The height and width of the output
        const float *x, const float *y,
        float *output,
        const int *activeIndices,
        const int *scatterMap, // [H, W, 3]
        const float *scale,
        int scaleB, int scaleC, int scaleH, int scaleW,
        const float *shift,
        int shiftB, int shiftC, int shiftH, int shiftW,
        ActivationType activationType, bool activationFirst) {
#pragma omp parallel for collapse(3)
    for (int bb = 0; bb < B; ++bb)
        for (int ib = 0; ib < numActive; ++ib)
            for (int cc = 0; cc < C; ++cc) {
                int biH = activeIndices[ib << 1];
                int biW = activeIndices[ib << 1 | 1];
                for (int intraBh = 0; intraBh < Ro; ++intraBh) {
                    int hh = biH + intraBh;
                    for (int intraBw = 0; intraBw < So; ++intraBw) {
                        int ww = biW + intraBw;
                        int index = (bb * numActive + ib) * C * Ro * So + cc * Ro * So + intraBh * So + intraBw;
                        if (hh < 0 || hh >= H || ww < 0 || ww >= W) {
                            output[index] = 0;
                            continue;
                        }
                        int scatterMapIndex = (hh * W + ww) * 3;
                        int bx = scatterMap[scatterMapIndex];
                        int p = bb * C * H * W + cc * H * W + hh * W + ww;
                        float z;
                        if (bx >= 0) {
                            int hx = scatterMap[scatterMapIndex + 1], wx = scatterMap[scatterMapIndex + 2];
                            z = x[(bb * numActive + bx) * C * Rx * Sx + cc * Rx * Sx + hx * Sx + wx];
                        } else
                            z = y[p];
                        if (!activationFirst) {
                            z = binary_op_array<MUL>(scale, z, scaleB, scaleC, scaleH, scaleW, bb, cc, hh, ww);
                            z = binary_op_array<ADD>(shift, z, shiftB, shiftC, shiftH, shiftW, bb, cc, hh, ww);
                        }
                        z = activation(activationType, z);
                        if (activationFirst) {
                            z = binary_op_array<MUL>(scale, z, scaleB, scaleC, scaleH, scaleW, bb, cc, hh, ww);
                            z = binary_op_array<ADD>(shift, z, shiftB, shiftC, shiftH, shiftW, bb, cc, hh, ww);
                        }
                        output[index] = z;
                    }
                }
            }
}

void get_scatter_map_cpu_kernel(
        int H, int W,
        int numActive, int R, int S,
        int offsetH, int offsetW,
        int strideH, int strideW,
        int *output,
        const int *activeIndices) {
#pragma omp parallel for
    for (int ib = 0; ib < numActive; ++ib) {
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
                auto p = 3 * (hh * W + ww);
                output[p] = ib;
                output[p + 1] = intraBh;
                output[p + 2] = intraBw;
            }
        }
    }
}

torch::Tensor scatter_gather_cpu(
        const torch::Tensor &x, // [B, C, bSizeH1, bSizeH2]
        const torch::Tensor &y, // [1, C, H, W]
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
    auto output = torch::empty({numActive, C, Ro, So}, options);
    auto xData = x.data_ptr<float>();
    auto yData = y.data_ptr<float>();
    auto outputData = output.data_ptr<float>();
    auto activeIndicesData = activeIndices.data_ptr<int>();
    auto scatterMapData = scatterMap.data_ptr<int>();

    const float *scaleData = nullptr;
    int scaleB = 0, scaleC = 0, scaleH = 0, scaleW = 0;
    if (scale.has_value()) {
        assert(broadcastable(y, scale.value()));
        scaleData = scale.value().data_ptr<float>();
        scaleB = scale.value().size(0);
        scaleC = scale.value().size(1);
        scaleH = scale.value().size(2);
        scaleW = scale.value().size(3);
    }

    const float *shiftData = nullptr;
    int shiftB = 0, shiftC = 0, shiftH = 0, shiftW = 0;
    if (shift.has_value()) {
        assert(broadcastable(y, shift.value()));
        shiftData = shift.value().data_ptr<float>();
        shiftB = shift.value().size(0);
        shiftC = shift.value().size(1);
        shiftH = shift.value().size(2);
        shiftW = shift.value().size(3);
    }

    const auto activationType = getActivationType(activationName);

    scatter_gather_cpu_kernel(
            numActive,
            B, C, H, W,
            Rx, Sx, Ro, So,
            xData, yData, outputData,
            activeIndicesData, scatterMapData,
            scaleData,
            scaleB, scaleC, scaleH, scaleW,
            shiftData,
            shiftB, shiftC, shiftH, shiftW,
            activationType, activationFirst);
    return output;
}

torch::Tensor get_scatter_map_cpu(
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

    get_scatter_map_cpu_kernel(
            H, W,
            numActive, R, S,
            offsetH, offsetW,
            strideH, strideW,
            scatterMap.data_ptr<int>(),
            activeIndices.data_ptr<int>());

    return scatterMap;
}
