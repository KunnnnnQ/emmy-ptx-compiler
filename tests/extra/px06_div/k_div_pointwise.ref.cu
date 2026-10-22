extern "C" __global__
__launch_bounds__(256) void k_div_pointwise(const float* add_c1, const float* b, const float* a, float* div) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in2__u0 = b[a0 * 2];
        float in0__u0 = add_c1[0];
        float in1__u0 = a[a0 * 2];
        float v2__u0 = in1__u0 / (in0__u0 + fabsf(in2__u0));
        float in2__u1 = b[a0 * 2 + 1];
        float in0__u1 = add_c1[0];
        float in1__u1 = a[a0 * 2 + 1];
        float v2__u1 = in1__u1 / (in0__u1 + fabsf(in2__u1));
        float2 _vs_v2__u0_39c58e90 = make_float2(v2__u0, v2__u1);
        *reinterpret_cast<float2*>(&div[a0 * 2]) = _vs_v2__u0_39c58e90;
}
