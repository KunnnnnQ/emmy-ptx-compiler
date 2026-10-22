extern "C" __global__
__launch_bounds__(128) void k_sum_1_reduce(const float* x, float* sum_1) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_blk = _gid / 128;
        int a1_co = (_gid / 32) % 4;
        int a0_ln = (_gid) % 32;
        float acc0 = 0.0f;
        for (int a1 = a1_co; a1 < 512; a1 += 4) {
            float in0 = x[(a0_blk * 32 + a0_ln) * 512 + a1];
            acc0 += in0;
        }
        __shared__ float acc0_smem[128];
        acc0_smem[a1_co * 32 + a0_ln] = acc0;
        __syncthreads();
        for (int s = 2; s > 0; s >>= 1) {
            if (a1_co < s) {
                float acc0 = acc0_smem[a1_co * 32 + a0_ln];
                float acc0__o = acc0_smem[(a1_co + s) * 32 + a0_ln];
                acc0 = acc0 + acc0__o;
                acc0_smem[a1_co * 32 + a0_ln] = acc0;
            }
            __syncthreads();
        }
        acc0 = acc0_smem[a0_ln];
        if (a1_co == 0) {
            sum_1[a0_blk * 32 + a0_ln] = acc0;
        }
}
