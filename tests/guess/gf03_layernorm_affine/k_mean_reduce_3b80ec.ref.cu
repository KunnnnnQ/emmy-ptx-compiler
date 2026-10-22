extern "C" __global__
__launch_bounds__(128) void k_mean_reduce_3b80ec(const float* mean_1_count, const float* mean_count, const float* add_c1, const float* x0, const float* x1, const float* x2, float* add_1) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 128;
        int a1_co = (_gid) % 128;
        float acc0 = 0.0f;
        for (int a1 = a1_co; a1 < 256; a1 += 128) {
            float in3 = x0[a0 * 256 + a1];
            acc0 += in3;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            acc0 = acc0 + acc0__o;
        }
        __shared__ float acc0_smem[4];
        if (lane == 0) {
            acc0_smem[warp] = acc0;
        }
        __syncthreads();
        for (int s = 2; s > 0; s >>= 1) {
            if (warp < s) {
                float acc0 = acc0_smem[warp];
                float acc0__o = acc0_smem[warp + s];
                acc0 = acc0 + acc0__o;
                acc0_smem[warp] = acc0;
            }
            __syncthreads();
        }
        acc0 = acc0_smem[0];
        float in2 = mean_count[0];
        float v2 = acc0 * (1.0f / in2);
        float acc1 = 0.0f;
        for (int a1 = 0; a1 < 256; a1++) {
            float in4 = x0[a0 * 256 + a1];
            float v3 = in4 - v2;
            float v4 = v3 * v3;
            acc1 += v4;
        }
        float in1 = mean_1_count[0];
        float in0 = add_c1[0];
        float v8 = 1.0f / sqrtf((in0 + (acc1 * (1.0f / in1))));
        for (int a3 = a1_co; a3 < 256; a3 += 128) {
            float in5 = x0[a0 * 256 + a3];
            float in6 = x1[a3];
            float in7 = x2[a3];
            float v12 = in7 + (in6 * (v8 * (in5 - v2)));
            add_1[a0 * 256 + a3] = v12;
        }
}
