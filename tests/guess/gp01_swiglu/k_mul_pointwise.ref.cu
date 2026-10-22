extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const float* silu_one, const float* x, const float* y, float* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in1__u0 = x[a0 * 4];
        float in0__u0 = silu_one[0];
        float in2__u0 = y[a0 * 4];
        float v5__u0 = in2__u0 * (in1__u0 * (1.0f / (in0__u0 + expf((0.0f - in1__u0)))));
        float in1__u1 = x[a0 * 4 + 1];
        float in0__u1 = silu_one[0];
        float in2__u1 = y[a0 * 4 + 1];
        float v5__u1 = in2__u1 * (in1__u1 * (1.0f / (in0__u1 + expf((0.0f - in1__u1)))));
        float in1__u2 = x[a0 * 4 + 2];
        float in0__u2 = silu_one[0];
        float in2__u2 = y[a0 * 4 + 2];
        float v5__u2 = in2__u2 * (in1__u2 * (1.0f / (in0__u2 + expf((0.0f - in1__u2)))));
        float in1__u3 = x[a0 * 4 + 3];
        float in0__u3 = silu_one[0];
        float in2__u3 = y[a0 * 4 + 3];
        float v5__u3 = in2__u3 * (in1__u3 * (1.0f / (in0__u3 + expf((0.0f - in1__u3)))));
        float4 _vs_v5__u0_1620d1da = make_float4(v5__u0, v5__u1, v5__u2, v5__u3);
        *reinterpret_cast<float4*>(&mul[a0 * 4]) = _vs_v5__u0_1620d1da;
}
