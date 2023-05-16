#include <metal_stdlib>
#include "common_mps.metal"

using namespace metal;

kernel void scatter_gather_mps_kernel(
        constant int &total, constant int &numActive,
        constant int &B, constant int &C, constant int &H, constant int &W,
        constant int &Rx, constant int &Sx, // The height and width of x
        constant int &Ro, constant int &So, // The height and width of the output
        device const float *x, device const float *y,
        device float *output,
        device const int *activeIndices,
        device const int *scatterMap, // [H, W, 3]
        device const float *scale,
        constant int &scaleB, constant int &scaleC, constant int &scaleH, constant int &scaleW,
        device const float *shift,
        constant int &shiftB, constant int &shiftC, constant int &shiftH, constant int &shiftW,
        constant ActivationType &activationType, constant bool &activationFirst,
        uint index [[thread_position_in_grid]]) {
    if (index >= total)
        return;
    int t = index;
    int intraBw = t % So;
    t /= So;
    int intraBh = t % Ro;
    t /= Ro;
    int cc = t % C;
    t /= C;
    int ib = t % numActive, bb = t / numActive;

    int biH = activeIndices[ib << 1];
    int hh = biH + intraBh;
    if (hh < 0 || hh >= H) {
        output[index] = 0;
        return;
    }
    int biW = activeIndices[ib << 1 | 1];
    int ww = biW + intraBw;
    if (ww < 0 || ww >= W) {
        output[index] = 0;
        return;
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
        z = binary_op_array_mps<MUL>(scale, z, scaleB, scaleC, scaleH, scaleW, bb, cc, hh, ww);
        z = binary_op_array_mps<ADD>(shift, z, shiftB, shiftC, shiftH, shiftW, bb, cc, hh, ww);
    }
    z = activation_mps(activationType, z);
    if (activationFirst) {
        z = binary_op_array_mps<MUL>(scale, z, scaleB, scaleC, scaleH, scaleW, bb, cc, hh, ww);
        z = binary_op_array_mps<ADD>(shift, z, shiftB, shiftC, shiftH, shiftW, bb, cc, hh, ww);
    }
    output[index] = z;
}

kernel void get_scatter_map_mps_kernel(
        constant int &total,
        constant int &H, constant int &W,
        constant int &R, constant int &S,
        constant int &offsetH, constant int &offsetW,
        constant int &strideH, constant int &strideW,
        device int *output,
        device const int *activeIndices,
        uint index [[thread_position_in_grid]]) {
    if (index >= total)
        return;
    int t = index;
    int intraBw = t % S;
    t /= S;
    int intraBh = t % R;
    t /= R;
    int ib = t;
    int biH = (offsetH + activeIndices[ib << 1]) / strideH;
    int hh = biH + intraBh;
    if (hh >= H)
        return;
    int biW = (offsetW + activeIndices[ib << 1 | 1]) / strideW;
    int ww = biW + intraBw;
    if (ww >= W)
        return;
    auto p = 3 * (hh * W + ww);
    output[p] = ib;
    output[p + 1] = intraBh;
    output[p + 2] = intraBw;
}