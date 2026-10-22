extern "C" __global__
__launch_bounds__(256) void k_sum_1(const float* sum_1__partial, float* sum_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        float acc0 = 0.0f;
        for (int _ksplit = 0; _ksplit < 2; _ksplit++) {
            float acc0__p = sum_1__partial[_ksplit];
            acc0 += acc0__p;
        }
        sum_1[0] = acc0;
    }
}
