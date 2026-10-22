extern "C" __global__
__launch_bounds__(256) void k_matmul_softmax_reduce_e7732a(const float* mul_c1, const float* q, const float* kt, const float* v, float* matmul_1) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 131072;
        int a1 = (_gid / 4096) % 32;
        int a6 = (_gid / 256) % 16;
        int a2_co = (_gid) % 256;
        float in0 = mul_c1[0];
        float acc3 = 0.0f;
        float acc5__sum = 0.0f;
        float acc1 = -1e+30f;
        float acc3__r1 = 0.0f;
        float acc5__sum__r1 = 0.0f;
        float acc1__r1 = -1e+30f;
        float acc3__r2 = 0.0f;
        float acc5__sum__r2 = 0.0f;
        float acc1__r2 = -1e+30f;
        float acc3__r3 = 0.0f;
        float acc5__sum__r3 = 0.0f;
        float acc1__r3 = -1e+30f;
        for (int a2 = a2_co; a2 < 48; a2 += 1024) {
            float acc0 = 0.0f;
            for (int a3 = 0; a3 < 16; a3++) {
                float in2 = q[a0 * 512 + a1 * 16 + a3];
                float in1 = kt[a0 * 768 + a3 * 48 + a2];
                float v0 = in1 * in2;
                acc0 += v0;
            }
            float v1 = acc0 * in0;
            float one = 1.0f;
            float v1__gn = fmaxf(acc1, v1);
            float v1__alpha = expf((acc1 - v1__gn));
            float v1__beta = expf((v1 - v1__gn));
            float v1__acc3_sa = acc3 * v1__alpha;
            float v1__acc3_sb = one * v1__beta;
            acc3 = v1__acc3_sa + v1__acc3_sb;
            float v1__acc5__sum_sa = acc5__sum * v1__alpha;
            float in7 = v[a0 * 768 + a2 * 16 + a6];
            float v1__acc5__sum_sb = in7 * v1__beta;
            acc5__sum = v1__acc5__sum_sa + v1__acc5__sum_sb;
            acc1 = fmaxf(acc1, v1);
            float acc0__r1 = 0.0f;
            for (int a3 = 0; a3 < 16; a3++) {
                float in2__r1 = q[a0 * 512 + a1 * 16 + a3];
                float in1__r1 = kt[a0 * 768 + a3 * 48 + (a2 + 256) % 48];
                float v0__r1 = in1__r1 * in2__r1;
                acc0__r1 += v0__r1;
            }
            float v1__r1 = acc0__r1 * in0;
            float v1__r1__id = -1e+30f;
            float v1__r1__mv = ((a2 + 256 < 48) ? (((float)(v1__r1))) : (((float)(v1__r1__id))));
            float one__r1 = 1.0f;
            float v1__gn__r1 = fmaxf(acc1__r1, v1__r1__mv);
            float v1__alpha__r1 = expf((acc1__r1 - v1__gn__r1));
            float v1__beta__r1 = expf((v1__r1__mv - v1__gn__r1));
            float v1__acc3_sa__r1 = acc3__r1 * v1__alpha__r1;
            float v1__acc3_sb__r1 = one__r1 * v1__beta__r1;
            acc3__r1 = v1__acc3_sa__r1 + v1__acc3_sb__r1;
            float v1__acc5__sum_sa__r1 = acc5__sum__r1 * v1__alpha__r1;
            float in7__r1 = v[a0 * 768 + (a2 + 256) % 48 * 16 + a6];
            float v1__acc5__sum_sb__r1 = in7__r1 * v1__beta__r1;
            acc5__sum__r1 = v1__acc5__sum_sa__r1 + v1__acc5__sum_sb__r1;
            acc1__r1 = fmaxf(acc1__r1, v1__r1__mv);
            float acc0__r2 = 0.0f;
            for (int a3 = 0; a3 < 16; a3++) {
                float in2__r2 = q[a0 * 512 + a1 * 16 + a3];
                float in1__r2 = kt[a0 * 768 + a3 * 48 + (a2 + 512) % 48];
                float v0__r2 = in1__r2 * in2__r2;
                acc0__r2 += v0__r2;
            }
            float v1__r2 = acc0__r2 * in0;
            float v1__r2__id = -1e+30f;
            float v1__r2__mv = ((a2 + 512 < 48) ? (((float)(v1__r2))) : (((float)(v1__r2__id))));
            float one__r2 = 1.0f;
            float v1__gn__r2 = fmaxf(acc1__r2, v1__r2__mv);
            float v1__alpha__r2 = expf((acc1__r2 - v1__gn__r2));
            float v1__beta__r2 = expf((v1__r2__mv - v1__gn__r2));
            float v1__acc3_sa__r2 = acc3__r2 * v1__alpha__r2;
            float v1__acc3_sb__r2 = one__r2 * v1__beta__r2;
            acc3__r2 = v1__acc3_sa__r2 + v1__acc3_sb__r2;
            float v1__acc5__sum_sa__r2 = acc5__sum__r2 * v1__alpha__r2;
            float in7__r2 = v[a0 * 768 + (a2 + 512) % 48 * 16 + a6];
            float v1__acc5__sum_sb__r2 = in7__r2 * v1__beta__r2;
            acc5__sum__r2 = v1__acc5__sum_sa__r2 + v1__acc5__sum_sb__r2;
            acc1__r2 = fmaxf(acc1__r2, v1__r2__mv);
            float acc0__r3 = 0.0f;
            for (int a3 = 0; a3 < 16; a3++) {
                float in2__r3 = q[a0 * 512 + a1 * 16 + a3];
                float in1__r3 = kt[a0 * 768 + a3 * 48 + (a2 + 768) % 48];
                float v0__r3 = in1__r3 * in2__r3;
                acc0__r3 += v0__r3;
            }
            float v1__r3 = acc0__r3 * in0;
            float v1__r3__id = -1e+30f;
            float v1__r3__mv = ((a2 + 768 < 48) ? (((float)(v1__r3))) : (((float)(v1__r3__id))));
            float one__r3 = 1.0f;
            float v1__gn__r3 = fmaxf(acc1__r3, v1__r3__mv);
            float v1__alpha__r3 = expf((acc1__r3 - v1__gn__r3));
            float v1__beta__r3 = expf((v1__r3__mv - v1__gn__r3));
            float v1__acc3_sa__r3 = acc3__r3 * v1__alpha__r3;
            float v1__acc3_sb__r3 = one__r3 * v1__beta__r3;
            acc3__r3 = v1__acc3_sa__r3 + v1__acc3_sb__r3;
            float v1__acc5__sum_sa__r3 = acc5__sum__r3 * v1__alpha__r3;
            float in7__r3 = v[a0 * 768 + (a2 + 768) % 48 * 16 + a6];
            float v1__acc5__sum_sb__r3 = in7__r3 * v1__beta__r3;
            acc5__sum__r3 = v1__acc5__sum_sa__r3 + v1__acc5__sum_sb__r3;
            acc1__r3 = fmaxf(acc1__r3, v1__r3__mv);
        }
        float acc1__r1__gn = fmaxf(acc1, acc1__r1);
        float acc1__r1__alpha = expf((acc1 - acc1__r1__gn));
        float acc1__r1__beta = expf((acc1__r1 - acc1__r1__gn));
        float acc1__r1__acc3_sa = acc1__r1__alpha * acc3;
        float acc1__r1__acc3_sb = acc1__r1__beta * acc3__r1;
        acc3 = acc1__r1__acc3_sa + acc1__r1__acc3_sb;
        float acc1__r1__acc5__sum_sa = acc1__r1__alpha * acc5__sum;
        float acc1__r1__acc5__sum_sb = acc1__r1__beta * acc5__sum__r1;
        acc5__sum = acc1__r1__acc5__sum_sa + acc1__r1__acc5__sum_sb;
        acc1 = fmaxf(acc1, acc1__r1);
        float acc1__r2__gn = fmaxf(acc1, acc1__r2);
        float acc1__r2__alpha = expf((acc1 - acc1__r2__gn));
        float acc1__r2__beta = expf((acc1__r2 - acc1__r2__gn));
        float acc1__r2__acc3_sa = acc1__r2__alpha * acc3;
        float acc1__r2__acc3_sb = acc1__r2__beta * acc3__r2;
        acc3 = acc1__r2__acc3_sa + acc1__r2__acc3_sb;
        float acc1__r2__acc5__sum_sa = acc1__r2__alpha * acc5__sum;
        float acc1__r2__acc5__sum_sb = acc1__r2__beta * acc5__sum__r2;
        acc5__sum = acc1__r2__acc5__sum_sa + acc1__r2__acc5__sum_sb;
        acc1 = fmaxf(acc1, acc1__r2);
        float acc1__r3__gn = fmaxf(acc1, acc1__r3);
        float acc1__r3__alpha = expf((acc1 - acc1__r3__gn));
        float acc1__r3__beta = expf((acc1__r3 - acc1__r3__gn));
        float acc1__r3__acc3_sa = acc1__r3__alpha * acc3;
        float acc1__r3__acc3_sb = acc1__r3__beta * acc3__r3;
        acc3 = acc1__r3__acc3_sa + acc1__r3__acc3_sb;
        float acc1__r3__acc5__sum_sa = acc1__r3__alpha * acc5__sum;
        float acc1__r3__acc5__sum_sb = acc1__r3__beta * acc5__sum__r3;
        acc5__sum = acc1__r3__acc5__sum_sa + acc1__r3__acc5__sum_sb;
        acc1 = fmaxf(acc1, acc1__r3);
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 16);
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 16);
            float acc5__sum__o = __shfl_xor_sync(__activemask(), acc5__sum, 16);
            float acc1__o__gn = fmaxf(acc1, acc1__o);
            float acc1__o__dg = acc1 - acc1__o__gn;
            float acc1__o__alpha = expf(acc1__o__dg);
            float acc1__o__dg_o = acc1__o - acc1__o__gn;
            float acc1__o__beta = expf(acc1__o__dg_o);
            float acc1__o__acc3_sa = acc1__o__alpha * acc3;
            float acc1__o__acc3_sb = acc1__o__beta * acc3__o;
            acc3 = acc1__o__acc3_sa + acc1__o__acc3_sb;
            float acc1__o__acc5__sum_sa = acc1__o__alpha * acc5__sum;
            float acc1__o__acc5__sum_sb = acc1__o__beta * acc5__sum__o;
            acc5__sum = acc1__o__acc5__sum_sa + acc1__o__acc5__sum_sb;
            acc1 = acc1__o__gn;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 8);
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 8);
            float acc5__sum__o = __shfl_xor_sync(__activemask(), acc5__sum, 8);
            float acc1__o__gn = fmaxf(acc1, acc1__o);
            float acc1__o__dg = acc1 - acc1__o__gn;
            float acc1__o__alpha = expf(acc1__o__dg);
            float acc1__o__dg_o = acc1__o - acc1__o__gn;
            float acc1__o__beta = expf(acc1__o__dg_o);
            float acc1__o__acc3_sa = acc1__o__alpha * acc3;
            float acc1__o__acc3_sb = acc1__o__beta * acc3__o;
            acc3 = acc1__o__acc3_sa + acc1__o__acc3_sb;
            float acc1__o__acc5__sum_sa = acc1__o__alpha * acc5__sum;
            float acc1__o__acc5__sum_sb = acc1__o__beta * acc5__sum__o;
            acc5__sum = acc1__o__acc5__sum_sa + acc1__o__acc5__sum_sb;
            acc1 = acc1__o__gn;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 4);
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 4);
            float acc5__sum__o = __shfl_xor_sync(__activemask(), acc5__sum, 4);
            float acc1__o__gn = fmaxf(acc1, acc1__o);
            float acc1__o__dg = acc1 - acc1__o__gn;
            float acc1__o__alpha = expf(acc1__o__dg);
            float acc1__o__dg_o = acc1__o - acc1__o__gn;
            float acc1__o__beta = expf(acc1__o__dg_o);
            float acc1__o__acc3_sa = acc1__o__alpha * acc3;
            float acc1__o__acc3_sb = acc1__o__beta * acc3__o;
            acc3 = acc1__o__acc3_sa + acc1__o__acc3_sb;
            float acc1__o__acc5__sum_sa = acc1__o__alpha * acc5__sum;
            float acc1__o__acc5__sum_sb = acc1__o__beta * acc5__sum__o;
            acc5__sum = acc1__o__acc5__sum_sa + acc1__o__acc5__sum_sb;
            acc1 = acc1__o__gn;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 2);
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 2);
            float acc5__sum__o = __shfl_xor_sync(__activemask(), acc5__sum, 2);
            float acc1__o__gn = fmaxf(acc1, acc1__o);
            float acc1__o__dg = acc1 - acc1__o__gn;
            float acc1__o__alpha = expf(acc1__o__dg);
            float acc1__o__dg_o = acc1__o - acc1__o__gn;
            float acc1__o__beta = expf(acc1__o__dg_o);
            float acc1__o__acc3_sa = acc1__o__alpha * acc3;
            float acc1__o__acc3_sb = acc1__o__beta * acc3__o;
            acc3 = acc1__o__acc3_sa + acc1__o__acc3_sb;
            float acc1__o__acc5__sum_sa = acc1__o__alpha * acc5__sum;
            float acc1__o__acc5__sum_sb = acc1__o__beta * acc5__sum__o;
            acc5__sum = acc1__o__acc5__sum_sa + acc1__o__acc5__sum_sb;
            acc1 = acc1__o__gn;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 1);
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 1);
            float acc5__sum__o = __shfl_xor_sync(__activemask(), acc5__sum, 1);
            float acc1__o__gn = fmaxf(acc1, acc1__o);
            float acc1__o__dg = acc1 - acc1__o__gn;
            float acc1__o__alpha = expf(acc1__o__dg);
            float acc1__o__dg_o = acc1__o - acc1__o__gn;
            float acc1__o__beta = expf(acc1__o__dg_o);
            float acc1__o__acc3_sa = acc1__o__alpha * acc3;
            float acc1__o__acc3_sb = acc1__o__beta * acc3__o;
            acc3 = acc1__o__acc3_sa + acc1__o__acc3_sb;
            float acc1__o__acc5__sum_sa = acc1__o__alpha * acc5__sum;
            float acc1__o__acc5__sum_sb = acc1__o__beta * acc5__sum__o;
            acc5__sum = acc1__o__acc5__sum_sa + acc1__o__acc5__sum_sb;
            acc1 = acc1__o__gn;
        }
        __shared__ float acc1_smem[8];
        __shared__ float acc3_smem[8];
        __shared__ float acc5__sum_smem[8];
        if (lane == 0) {
            acc1_smem[warp] = acc1;
            acc3_smem[warp] = acc3;
            acc5__sum_smem[warp] = acc5__sum;
        }
        __syncthreads();
        for (int s = 4; s > 0; s >>= 1) {
            if (warp < s) {
                float acc1 = acc1_smem[warp];
                float acc1__o = acc1_smem[warp + s];
                float acc3 = acc3_smem[warp];
                float acc3__o = acc3_smem[warp + s];
                float acc5__sum = acc5__sum_smem[warp];
                float acc5__sum__o = acc5__sum_smem[warp + s];
                float acc1__o__gn = fmaxf(acc1, acc1__o);
                float acc1__o__dg = acc1 - acc1__o__gn;
                float acc1__o__alpha = expf(acc1__o__dg);
                float acc1__o__dg_o = acc1__o - acc1__o__gn;
                float acc1__o__beta = expf(acc1__o__dg_o);
                float acc1__o__acc3_sa = acc1__o__alpha * acc3;
                float acc1__o__acc3_sb = acc1__o__beta * acc3__o;
                acc3 = acc1__o__acc3_sa + acc1__o__acc3_sb;
                float acc1__o__acc5__sum_sa = acc1__o__alpha * acc5__sum;
                float acc1__o__acc5__sum_sb = acc1__o__beta * acc5__sum__o;
                acc5__sum = acc1__o__acc5__sum_sa + acc1__o__acc5__sum_sb;
                acc1 = acc1__o__gn;
                acc1_smem[warp] = acc1;
                acc3_smem[warp] = acc3;
                acc5__sum_smem[warp] = acc5__sum;
            }
            __syncthreads();
        }
        acc1 = acc1_smem[0];
        acc3 = acc3_smem[0];
        acc5__sum = acc5__sum_smem[0];
        if (a2_co == 0) {
            float acc5 = acc5__sum * (1.0f / acc3);
            matmul_1[a0 * 512 + a1 * 16 + a6] = acc5;
        }
}
