extern "C" __global__
__launch_bounds__(32) void k_linear_d84d36(const float* b, const float* x, const float* w, float* linear) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 2048;
        int a1 = (_gid / 32) % 64;
        int a2_co = (_gid) % 32;
        float acc0 = 0.0f;
        for (int a2 = a2_co; a2 < 128; a2 += 32) {
            float in1 = x[a0 * 128 + a2];
            float in0 = w[a1 * 128 + a2];
            float v0 = in0 * in1;
            acc0 += v0;
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
        if (a2_co == 0) {
            float in2 = b[a1];
            float v1 = acc0 + in2;
            linear[a0 * 64 + a1] = v1;
        }
}
