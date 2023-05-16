#include <metal_stdlib>
#include "common_mps.metal"

using namespace metal;

kernel void scatter_mps_kernel(
        constant int &total, constant int &numActive,
        constant int &B, constant int &C, constant int &H, constant int &W,
        constant int &R, constant int &S,
        constant int &offsetH, constant int &offsetW,
        constant int &strideH, constant int &strideW,
        device const float *x, device float *output,
        device const int *activeIndices,
        device const float *residual,
        constant int &residualB, constant int &residualC, constant int &residualH, constant int &residualW,
        uint index [[thread_position_in_grid]]) {
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
    z = binary_op_array_mps<ADD>(
            residual, z,
            residualB, residualC, residualH, residualW,
            bb, cc, hh, ww);
    output[p] = z;
}

kernel void calibrate_residual_mps_kernel(
        constant int &total, constant int &numActive,
        constant int &B, constant int &C, constant int &H, constant int &W,
        constant int &R, constant int &S,
        device const float *x, device const float *y,
        device float *output,
        device const int *activeIndices,
        uint index [[thread_position_in_grid]]) {
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