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
__launch_bounds__(256) void k_matmul(const float* x1, const float* x0, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a1_ks = _gid / 24576;
        int _um_b = (_gid / 24576) % 1;
        int a0_b = (_gid / 256) % 96;
        int _um_u = (_gid / 32) % 8;
        int a0_u = (_gid) % 32;
        float acc0__c0_0 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c2_0 = 0.0f;
        float acc0__c3_0 = 0.0f;
        __shared__ __align__(16) float _a_smem[512];
        __shared__ __align__(16) float _b_smem[512];
        for (int _ks = 0; _ks < 16; _ks += 16) {
            for (int _fa = _um_u * 32 + a0_u; _fa < 128; _fa += 256) {
                emmy_cp_async_cg(&_a_smem[_fa * 4], &x0[_a1_ks * 16 + (_ks + 4 * (_fa % 4))]);
            }
            for (int _fb = _um_u * 32 + a0_u; _fb < 128; _fb += 256) {
                emmy_cp_async_cg(&_b_smem[_fb * 4], &x1[(_a1_ks * 16 + (_ks + _fb / 8)) * 3072 + (a0_b * 32 + 4 * (_fb % 8))]);
            }
            emmy_cp_async_commit();
            emmy_cp_async_wait<0>();
            __syncthreads();
            #pragma unroll
            for (int _ki = 0; _ki < 16; _ki++) {
                float in1__c0_0 = _b_smem[_ki * 32 + (a0_b * 32 + a0_u - a0_b * 32)];
                float in0__c0_0 = _a_smem[(_um_b * 32 + _um_u * 4 - _um_b * 32) * 16 + _ki];
                float acc0__v__c0_0 = in1__c0_0 * in0__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
                float in0__c1_0 = _a_smem[(_um_b * 32 + _um_u * 4 + 1 - _um_b * 32) * 16 + _ki];
                float acc0__v__c1_0 = in1__c0_0 * in0__c1_0;
                acc0__c1_0 += acc0__v__c1_0;
                float in0__c2_0 = _a_smem[(_um_b * 32 + _um_u * 4 + 2 - _um_b * 32) * 16 + _ki];
                float acc0__v__c2_0 = in1__c0_0 * in0__c2_0;
                acc0__c2_0 += acc0__v__c2_0;
                float in0__c3_0 = _a_smem[(_um_b * 32 + _um_u * 4 + 3 - _um_b * 32) * 16 + _ki];
                float acc0__v__c3_0 = in1__c0_0 * in0__c3_0;
                acc0__c3_0 += acc0__v__c3_0;
            }
            __syncthreads();
        }
        if (_um_b * 32 + _um_u * 4 + 0 < 1) {
            atomicAdd(&matmul[a0_b * 32 + a0_u], acc0__c0_0);
        }
        if (_um_b * 32 + _um_u * 4 + 1 < 1) {
            atomicAdd(&matmul[a0_b * 32 + a0_u], acc0__c1_0);
        }
        if (_um_b * 32 + _um_u * 4 + 2 < 1) {
            atomicAdd(&matmul[a0_b * 32 + a0_u], acc0__c2_0);
        }
        if (_um_b * 32 + _um_u * 4 + 3 < 1) {
            atomicAdd(&matmul[a0_b * 32 + a0_u], acc0__c3_0);
        }
}
