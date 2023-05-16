#include <metal_stdlib>
#include "common_mps.metal"

using namespace metal;

kernel void gather_mps_kernel(
        constant int &total, constant int &numActive,
        constant int &B, constant int &C, constant int &H, constant int &W,
        constant int &R, constant int &S,
        device const float *x,
        device float *output,
        device const int *activeIndices,
        device const float *scale,
        constant int &scaleB, constant int &scaleC, constant int &scaleH, constant int &scaleW,
        device const float *shift,
        constant int &shiftB, constant int &shiftC, constant int &shiftH, constant int &shiftW,
        constant ActivationType &activationType,
        constant bool &activationFirst,
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
    auto p = bb * C * H * W + cc * H * W + hh * W + ww;
    auto z = x[p];
    if (!activationFirst) {
        z = binary_op_array_mps<MUL>(
                scale, z,
                scaleB, scaleC, scaleH, scaleW,
                bb, cc, hh, ww);
        z = binary_op_array_mps<ADD>(
                shift, z,
                shiftB, shiftC, shiftH, shiftW,
                bb, cc, hh, ww);
    }
    z = activation_mps(activationType, z);
    if (activationFirst) {
        z = binary_op_array_mps<MUL>(
                scale, z,
                scaleB, scaleC, scaleH, scaleW,
                bb, cc, hh, ww);
        z = binary_op_array_mps<ADD>(
                shift, z,
                shiftB, shiftC, shiftH, shiftW,
                bb, cc, hh, ww);
    }
    output[index] = z;
}