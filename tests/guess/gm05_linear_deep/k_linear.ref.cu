extern "C" __global__
__launch_bounds__(256) void k_linear(const float* x, const float* w, float* linear) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a1_ks = _gid / 128;
        int _um_b = (_gid / 128) % 1;
        int a0_b = (_gid) % 128;
        int _um_u = (_gid) % 1;
        int a0_u = (_gid) % 1;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        float acc0__c0_2 = 0.0f;
        float acc0__c0_3 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c1_1 = 0.0f;
        float acc0__c1_2 = 0.0f;
        float acc0__c1_3 = 0.0f;
        for (int a1 = 0; a1 < 2048; a1++) {
            float in0__ar0 = x[_a1_ks * 2048 + a1];
            float in1__bc0 = w[(a0_b * 4 + a0_u * 4) * 4096 + (_a1_ks * 2048 + a1)];
            float v0__c0_0 = in0__ar0 * in1__bc0;
            acc0__c0_0 += v0__c0_0;
            float in1__bc1 = w[(a0_b * 4 + a0_u * 4 + 1) * 4096 + (_a1_ks * 2048 + a1)];
            float v0__c0_1 = in0__ar0 * in1__bc1;
            acc0__c0_1 += v0__c0_1;
            float in1__bc2 = w[(a0_b * 4 + a0_u * 4 + 2) * 4096 + (_a1_ks * 2048 + a1)];
            float v0__c0_2 = in0__ar0 * in1__bc2;
            acc0__c0_2 += v0__c0_2;
            float in1__bc3 = w[(a0_b * 4 + a0_u * 4 + 3) * 4096 + (_a1_ks * 2048 + a1)];
            float v0__c0_3 = in0__ar0 * in1__bc3;
            acc0__c0_3 += v0__c0_3;
            float in0__ar1 = x[_a1_ks * 2048 + a1];
            float v0__c1_0 = in0__ar1 * in1__bc0;
            acc0__c1_0 += v0__c1_0;
            float v0__c1_1 = in0__ar1 * in1__bc1;
            acc0__c1_1 += v0__c1_1;
            float v0__c1_2 = in0__ar1 * in1__bc2;
            acc0__c1_2 += v0__c1_2;
            float v0__c1_3 = in0__ar1 * in1__bc3;
            acc0__c1_3 += v0__c1_3;
        }
        if (_um_b * 2 + _um_u * 2 + 0 < 1) {
            atomicAdd(&linear[a0_b * 4 + a0_u * 4], acc0__c0_0);
        }
        if (_um_b * 2 + _um_u * 2 + 0 < 1) {
            atomicAdd(&linear[a0_b * 4 + a0_u * 4 + 1], acc0__c0_1);
        }
        if (_um_b * 2 + _um_u * 2 + 0 < 1) {
            atomicAdd(&linear[a0_b * 4 + a0_u * 4 + 2], acc0__c0_2);
        }
        if (_um_b * 2 + _um_u * 2 + 0 < 1) {
            atomicAdd(&linear[a0_b * 4 + a0_u * 4 + 3], acc0__c0_3);
        }
        if (_um_b * 2 + _um_u * 2 + 1 < 1) {
            atomicAdd(&linear[a0_b * 4 + a0_u * 4], acc0__c1_0);
        }
        if (_um_b * 2 + _um_u * 2 + 1 < 1) {
            atomicAdd(&linear[a0_b * 4 + a0_u * 4 + 1], acc0__c1_1);
        }
        if (_um_b * 2 + _um_u * 2 + 1 < 1) {
            atomicAdd(&linear[a0_b * 4 + a0_u * 4 + 2], acc0__c1_2);
        }
        if (_um_b * 2 + _um_u * 2 + 1 < 1) {
            atomicAdd(&linear[a0_b * 4 + a0_u * 4 + 3], acc0__c1_3);
        }
}
