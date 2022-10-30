#include "../common.cpp"
#include <cuda_runtime.h>

const int threads = 512;

template<OpType opType>
__device__ __forceinline__ float binary_op_cuda(float a, float b) {
    if (opType == ADD)
        return a + b;
    else if (opType == MUL)
        return a * b;
    else {
        __builtin_assume(false);
        return 0;
    }
}

template<OpType opType>
__device__ __forceinline__ float binary_op_array_cuda(
        const float *__restrict__ x, float y,
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
        return binary_op_cuda<opType>(x[p], y);
    }
}

__device__ __forceinline__ float activation_cuda(ActivationType activationType, float z) {
    if (activationType == IDENTITY)
        return z;
    else if (activationType == SWISH)
        return z / (1.0 + exp(-z));
    else {
        __builtin_assume(false);
        return IDENTITY;
    }
}
