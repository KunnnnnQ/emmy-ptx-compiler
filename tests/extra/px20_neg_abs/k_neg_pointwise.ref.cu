extern "C" __global__
__launch_bounds__(256) void k_neg_pointwise(const float* x, float* neg) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 750) {
        int a0 = _gid;
        float4 _v_in0__u0 = *reinterpret_cast<const float4*>(&x[a0 * 4]);
        float in0__u0 = _v_in0__u0.x;
        float in0__u1 = _v_in0__u0.y;
        float in0__u2 = _v_in0__u0.z;
        float in0__u3 = _v_in0__u0.w;
        float v1__u0 = 0.0f - fabsf(in0__u0);
        float v1__u1 = 0.0f - fabsf(in0__u1);
        float v1__u2 = 0.0f - fabsf(in0__u2);
        float v1__u3 = 0.0f - fabsf(in0__u3);
        float4 _vs_v1__u0_accb7565 = make_float4(v1__u0, v1__u1, v1__u2, v1__u3);
        *reinterpret_cast<float4*>(&neg[a0 * 4]) = _vs_v1__u0_accb7565;
    }
}
