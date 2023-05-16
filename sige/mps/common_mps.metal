#include <metal_math>
using namespace metal;

enum OpType {
    ADD,
    MUL
};

enum ActivationType {
    IDENTITY,
    SWISH
};

template<OpType opType>
inline float binary_op_mps(float a, float b) {
    if (opType == ADD)
        return a + b;
    else if (opType == MUL)
        return a * b;
    else return 0;
}

template<OpType opType>
inline float binary_op_array_mps(
        device const float *x, float y,
        int B, int C, int H, int W,
        int b, int c, int h, int w
) {
    if (B == 0 && C == 0 && H == 0 && W == 0) return y;
    else {
        int p = 0;
        if (W > 1) p = w;
        if (H > 1) p += h * W;
        if (C > 1) p += c * H * W;
        if (B > 1) p += b * C * H * W;
        return binary_op_mps<opType>(x[p], y);
    }
}

inline float activation_mps(ActivationType activationType, float z) {
    if (activationType == IDENTITY)
        return z;
    else if (activationType == SWISH)
        return z / (1.0 + exp(-z));
    else return IDENTITY;
}