#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_silu_pointwise(const float* silu_one, const __half* x, __half* silu) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        __half in1__u0 = x[a0 * 4];
        float v0__u0 = __half2float(in1__u0);
        float in0__u0 = silu_one[0];
        __half v5__u0 = __float2half(v0__u0 * (1.0f / (in0__u0 + expf((0.0f - v0__u0)))));
        __half in1__u1 = x[a0 * 4 + 1];
        float v0__u1 = __half2float(in1__u1);
        float in0__u1 = silu_one[0];
        __half v5__u1 = __float2half(v0__u1 * (1.0f / (in0__u1 + expf((0.0f - v0__u1)))));
        __half in1__u2 = x[a0 * 4 + 2];
        float v0__u2 = __half2float(in1__u2);
        float in0__u2 = silu_one[0];
        __half v5__u2 = __float2half(v0__u2 * (1.0f / (in0__u2 + expf((0.0f - v0__u2)))));
        __half in1__u3 = x[a0 * 4 + 3];
        float v0__u3 = __half2float(in1__u3);
        float in0__u3 = silu_one[0];
        __half v5__u3 = __float2half(v0__u3 * (1.0f / (in0__u3 + expf((0.0f - v0__u3)))));
        __half _vs_v5__u0_e39eaa86[4] = { v5__u0, v5__u1, v5__u2, v5__u3 };
        *reinterpret_cast<uint2*>(&silu[a0 * 4]) = *reinterpret_cast<const uint2*>(_vs_v5__u0_e39eaa86);
}
