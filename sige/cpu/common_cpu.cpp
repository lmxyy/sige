#include "../common.cpp"

template<OpType opType>
inline float binary_op(float a, float b) {
    if (opType == ADD)
        return a + b;
    else if (opType == MUL)
        return a * b;
    else __builtin_unreachable();
}

template<OpType opType>
inline float binary_op_array(
        const float *x, float y,
        int B, int C, int H, int W,
        int b, int c, int h, int w
) {
    if (x == nullptr) return y;
    else {
        int p = 0;
        if (W > 1) p = w;
        if (H > 1) p += h * W;
        if (C > 1) p += c * H * W;
        if (B > 1) p += b * C * H * W;
        return binary_op<opType>(x[p], y);
    }
}

inline float activation(ActivationType activationType, float z) {
    if (activationType == IDENTITY)
        return z;
    else if (activationType == SWISH)
        return z / (1.0 + exp(-z));
    else __builtin_unreachable();
}
