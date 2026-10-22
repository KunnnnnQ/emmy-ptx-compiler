extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* add_c1, const float* x, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 333) {
        int a0 = _gid;
        float in0__u0 = add_c1[0];
        float in1__u0 = x[a0 * 3];
        float v0__u0 = in0__u0 + in1__u0;
        float in0__u1 = add_c1[0];
        float in1__u1 = x[a0 * 3 + 1];
        float v0__u1 = in0__u1 + in1__u1;
        float in0__u2 = add_c1[0];
        float in1__u2 = x[a0 * 3 + 2];
        float v0__u2 = in0__u2 + in1__u2;
        add[a0 * 3] = v0__u0;
        add[a0 * 3 + 1] = v0__u1;
        add[a0 * 3 + 2] = v0__u2;
    }
}
