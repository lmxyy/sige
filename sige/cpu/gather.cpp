#include "common_cpu.cpp"
#include <torch/extension.h>

void gather_cpu_kernel(
        int numActive,
        int B, int C, int H, int W,
        int R, int S,
        const float *x,
        float *output,
        const int *activeIndices,
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
                for (int intraBh = 0; intraBh < R; ++intraBh) {
                    int hh = biH + intraBh;
                    for (int intraBw = 0; intraBw < S; ++intraBw) {
                        int ww = biW + intraBw;
                        int index = (bb * numActive + ib) * C * R * S + cc * R * S + intraBh * S + intraBw;
                        if (hh < 0 || hh >= H || ww < 0 || ww >= W) {
                            output[index] = 0;
                            continue;
                        }
                        int p = bb * C * H * W + cc * H * W + hh * W + ww;
                        auto z = x[p];
                        if (!activationFirst) {
                            z = binary_op_array<MUL>(
                                    scale, z,
                                    scaleB, scaleC, scaleH, scaleW,
                                    bb, cc, hh, ww);
                            z = binary_op_array<ADD>(
                                    shift, z,
                                    shiftB, shiftC, shiftH, shiftW,
                                    bb, cc, hh, ww);
                        }
                        z = activation(activationType, z);
                        if (activationFirst) {
                            z = binary_op_array<MUL>(
                                    scale, z,
                                    scaleB, scaleC, scaleH, scaleW,
                                    bb, cc, hh, ww);
                            z = binary_op_array<ADD>(
                                    shift, z,
                                    shiftB, shiftC, shiftH, shiftW,
                                    bb, cc, hh, ww);
                        }
                        output[index] = z;
                    }
                }
            }
}

torch::Tensor gather_cpu(
        const torch::Tensor &x,
        int bSizeH, int bSizeW,
        const torch::Tensor &activeIndices,
        const torch::optional<torch::Tensor> &scale,
        const torch::optional<torch::Tensor> &shift,
        const std::string &activationName = std::string("identity"),
        bool activationFirst = false) {
    const int R = bSizeH, S = bSizeW;
    const int numActive = activeIndices.size(0);
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device()).requires_grad(false);
    auto xData = x.data_ptr<float>();
    const auto activeIndicesData = activeIndices.data_ptr<int>();

    const int B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    auto output = torch::empty({B * numActive, C, R, S}, options);
    auto outputData = output.data_ptr<float>();

    const float *scaleData = nullptr;
    int scaleB = 0, scaleC = 0, scaleH = 0, scaleW = 0;
    if (scale.has_value()) {
        assert(broadcastable(x, scale.value()));
        scaleData = scale.value().data_ptr<float>();
        scaleB = scale.value().size(0);
        scaleC = scale.value().size(1);
        scaleH = scale.value().size(2);
        scaleW = scale.value().size(3);
    }

    const float *shiftData = nullptr;
    int shiftB = 0, shiftC = 0, shiftH = 0, shiftW = 0;
    if (shift.has_value()) {
        assert(broadcastable(x, shift.value()));
        shiftData = shift.value().data_ptr<float>();
        shiftB = shift.value().size(0);
        shiftC = shift.value().size(1);
        shiftH = shift.value().size(2);
        shiftW = shift.value().size(3);
    }

    const auto activationType = getActivationType(activationName);

    gather_cpu_kernel(
            numActive,
            B, C, H, W,
            R, S,
            xData, outputData, activeIndicesData,
            scaleData,
            scaleB, scaleC, scaleH, scaleW,
            shiftData,
            shiftB, shiftC, shiftH, shiftW,
            activationType, activationFirst);

    return output;
}
