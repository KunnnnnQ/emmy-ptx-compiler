static __device__ __forceinline__ void emmy_cp_async_cg(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(gmem) : "memory");
}

template <int Bytes>
static __device__ __forceinline__ void emmy_cp_async_ca(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(addr), "l"(gmem), "n"(Bytes) : "memory");
}

static __device__ __forceinline__ void emmy_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
static __device__ __forceinline__ void emmy_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

extern "C" __global__
__launch_bounds__(128) void k_matmul_c60c8d(const float* x1, const float* x0, float* relu) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 512;
        int a1_b = (_gid / 128) % 4;
        int a0_u = (_gid / 16) % 8;
        int a1_u = (_gid) % 16;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c1_1 = 0.0f;
        float acc0__c2_0 = 0.0f;
        float acc0__c2_1 = 0.0f;
        float acc0__c3_0 = 0.0f;
        float acc0__c3_1 = 0.0f;
        __shared__ __align__(16) float _a_smem[2048];
        __shared__ __align__(16) float _b_smem[2048];
        for (int _ks = 0; _ks < 64; _ks += 64) {
            emmy_cp_async_cg(&_a_smem[0 + ((a0_u * 16 + a1_u) * 4 / 64) * 64 + ((a0_u * 16 + a1_u) * 4 % 64)], &x0[(a0_b * 32 + (a0_u * 16 + a1_u) / 16) * 64 + (_ks + 4 * ((a0_u * 16 + a1_u) % 16))]);
            emmy_cp_async_cg(&_a_smem[512 + ((a0_u * 16 + a1_u) * 4 / 64) * 64 + ((a0_u * 16 + a1_u) * 4 % 64)], &x0[(a0_b * 32 + ((a0_u * 16 + a1_u) / 16 + 8)) * 64 + (_ks + 4 * ((a0_u * 16 + a1_u) % 16))]);
            emmy_cp_async_cg(&_a_smem[1024 + ((a0_u * 16 + a1_u) * 4 / 64) * 64 + ((a0_u * 16 + a1_u) * 4 % 64)], &x0[(a0_b * 32 + ((a0_u * 16 + a1_u) / 16 + 16)) * 64 + (_ks + 4 * ((a0_u * 16 + a1_u) % 16))]);
            emmy_cp_async_cg(&_a_smem[1536 + ((a0_u * 16 + a1_u) * 4 / 64) * 64 + ((a0_u * 16 + a1_u) * 4 % 64)], &x0[(a0_b * 32 + ((a0_u * 16 + a1_u) / 16 + 24)) * 64 + (_ks + 4 * ((a0_u * 16 + a1_u) % 16))]);
            emmy_cp_async_cg(&_b_smem[0 + ((a0_u * 16 + a1_u) * 4 / 32) * 32 + ((a0_u * 16 + a1_u) * 4 % 32)], &x1[(_ks + (a0_u * 16 + a1_u) / 8) * 128 + (a1_b * 32 + 4 * ((a0_u * 16 + a1_u) % 8))]);
            emmy_cp_async_cg(&_b_smem[512 + ((a0_u * 16 + a1_u) * 4 / 32) * 32 + ((a0_u * 16 + a1_u) * 4 % 32)], &x1[(_ks + ((a0_u * 16 + a1_u) / 8 + 16)) * 128 + (a1_b * 32 + 4 * ((a0_u * 16 + a1_u) % 8))]);
            emmy_cp_async_cg(&_b_smem[1024 + ((a0_u * 16 + a1_u) * 4 / 32) * 32 + ((a0_u * 16 + a1_u) * 4 % 32)], &x1[(_ks + ((a0_u * 16 + a1_u) / 8 + 32)) * 128 + (a1_b * 32 + 4 * ((a0_u * 16 + a1_u) % 8))]);
            emmy_cp_async_cg(&_b_smem[1536 + ((a0_u * 16 + a1_u) * 4 / 32) * 32 + ((a0_u * 16 + a1_u) * 4 % 32)], &x1[(_ks + ((a0_u * 16 + a1_u) / 8 + 48)) * 128 + (a1_b * 32 + 4 * ((a0_u * 16 + a1_u) % 8))]);
            emmy_cp_async_commit();
            emmy_cp_async_wait<0>();
            __syncthreads();
            #pragma unroll
            for (int _ki = 0; _ki < 64; _ki++) {
                float in1__c0_0 = _b_smem[_ki * 32 + (a1_b * 32 + a1_u * 2 - a1_b * 32)];
                float in0__c0_0 = _a_smem[(a0_b * 32 + a0_u * 4 - a0_b * 32) * 64 + _ki];
                float acc0__v__c0_0 = in1__c0_0 * in0__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
                float in1__c0_1 = _b_smem[_ki * 32 + (a1_b * 32 + a1_u * 2 + 1 - a1_b * 32)];
                float acc0__v__c0_1 = in1__c0_1 * in0__c0_0;
                acc0__c0_1 += acc0__v__c0_1;
                float in0__c1_0 = _a_smem[(a0_b * 32 + a0_u * 4 + 1 - a0_b * 32) * 64 + _ki];
                float acc0__v__c1_0 = in1__c0_0 * in0__c1_0;
                acc0__c1_0 += acc0__v__c1_0;
                float acc0__v__c1_1 = in1__c0_1 * in0__c1_0;
                acc0__c1_1 += acc0__v__c1_1;
                float in0__c2_0 = _a_smem[(a0_b * 32 + a0_u * 4 + 2 - a0_b * 32) * 64 + _ki];
                float acc0__v__c2_0 = in1__c0_0 * in0__c2_0;
                acc0__c2_0 += acc0__v__c2_0;
                float acc0__v__c2_1 = in1__c0_1 * in0__c2_0;
                acc0__c2_1 += acc0__v__c2_1;
                float in0__c3_0 = _a_smem[(a0_b * 32 + a0_u * 4 + 3 - a0_b * 32) * 64 + _ki];
                float acc0__v__c3_0 = in1__c0_0 * in0__c3_0;
                acc0__c3_0 += acc0__v__c3_0;
                float acc0__v__c3_1 = in1__c0_1 * in0__c3_0;
                acc0__c3_1 += acc0__v__c3_1;
            }
            __syncthreads();
        }
        float v1__c0_0 = fmaxf(0.0f, acc0__c0_0);
        relu[(a0_b * 32 + a0_u * 4) * 128 + (a1_b * 32 + a1_u * 2)] = v1__c0_0;
        float v1__c0_1 = fmaxf(0.0f, acc0__c0_1);
        relu[(a0_b * 32 + a0_u * 4) * 128 + (a1_b * 32 + a1_u * 2 + 1)] = v1__c0_1;
        float v1__c1_0 = fmaxf(0.0f, acc0__c1_0);
        relu[(a0_b * 32 + a0_u * 4 + 1) * 128 + (a1_b * 32 + a1_u * 2)] = v1__c1_0;
        float v1__c1_1 = fmaxf(0.0f, acc0__c1_1);
        relu[(a0_b * 32 + a0_u * 4 + 1) * 128 + (a1_b * 32 + a1_u * 2 + 1)] = v1__c1_1;
        float v1__c2_0 = fmaxf(0.0f, acc0__c2_0);
        relu[(a0_b * 32 + a0_u * 4 + 2) * 128 + (a1_b * 32 + a1_u * 2)] = v1__c2_0;
        float v1__c2_1 = fmaxf(0.0f, acc0__c2_1);
        relu[(a0_b * 32 + a0_u * 4 + 2) * 128 + (a1_b * 32 + a1_u * 2 + 1)] = v1__c2_1;
        float v1__c3_0 = fmaxf(0.0f, acc0__c3_0);
        relu[(a0_b * 32 + a0_u * 4 + 3) * 128 + (a1_b * 32 + a1_u * 2)] = v1__c3_0;
        float v1__c3_1 = fmaxf(0.0f, acc0__c3_1);
        relu[(a0_b * 32 + a0_u * 4 + 3) * 128 + (a1_b * 32 + a1_u * 2 + 1)] = v1__c3_1;
}
