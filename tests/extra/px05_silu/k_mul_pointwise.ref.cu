extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const float* x, float* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float4 _v_in0__u0 = *reinterpret_cast<const float4*>(&x[a0 * 4]);
        float in0__u0 = _v_in0__u0.x;
        float in0__u1 = _v_in0__u0.y;
        float in0__u2 = _v_in0__u0.z;
        float in0__u3 = _v_in0__u0.w;
        float v1__u0 = in0__u0 * (1.0f / (1.0f + expf(0.0f - in0__u0)));
        float v1__u1 = in0__u1 * (1.0f / (1.0f + expf(0.0f - in0__u1)));
        float v1__u2 = in0__u2 * (1.0f / (1.0f + expf(0.0f - in0__u2)));
        float v1__u3 = in0__u3 * (1.0f / (1.0f + expf(0.0f - in0__u3)));
        float4 _vs_v1__u0_932c6ce9 = make_float4(v1__u0, v1__u1, v1__u2, v1__u3);
        *reinterpret_cast<float4*>(&mul[a0 * 4]) = _vs_v1__u0_932c6ce9;
}
