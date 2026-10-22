extern "C" __global__
__launch_bounds__(256) void k_linear_af43cf(const float* x, const float* w, float* linear) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 108) {
        int a0_b = _gid / 36;
        int a1_b = (_gid) % 36;
        int a0_u = (_gid) % 1;
        int a1_u = (_gid) % 1;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c1_1 = 0.0f;
        for (int a2 = 0; a2 < 96; a2++) {
            float in1__ar0 = x[(a0_b * 2 + a0_u * 2) % 5 * 96 + a2];
            float in0__bc0 = w[(a1_b * 2 + a1_u * 2) * 96 + a2];
            float v0__c0_0 = in0__bc0 * in1__ar0;
            acc0__c0_0 += v0__c0_0;
            float in0__bc1 = w[(a1_b * 2 + a1_u * 2 + 1) * 96 + a2];
            float v0__c0_1 = in0__bc1 * in1__ar0;
            acc0__c0_1 += v0__c0_1;
            float in1__ar1 = x[(a0_b * 2 + a0_u * 2 + 1) % 5 * 96 + a2];
            float v0__c1_0 = in0__bc0 * in1__ar1;
            acc0__c1_0 += v0__c1_0;
            float v0__c1_1 = in0__bc1 * in1__ar1;
            acc0__c1_1 += v0__c1_1;
        }
        if (a0_b * 2 + a0_u * 2 + 0 < 5) {
            linear[(a0_b * 2 + a0_u * 2) % 5 * 72 + (a1_b * 2 + a1_u * 2)] = acc0__c0_0;
        }
        if (a0_b * 2 + a0_u * 2 + 0 < 5) {
            linear[(a0_b * 2 + a0_u * 2) % 5 * 72 + (a1_b * 2 + a1_u * 2 + 1)] = acc0__c0_1;
        }
        if (a0_b * 2 + a0_u * 2 + 1 < 5) {
            linear[(a0_b * 2 + a0_u * 2 + 1) % 5 * 72 + (a1_b * 2 + a1_u * 2)] = acc0__c1_0;
        }
        if (a0_b * 2 + a0_u * 2 + 1 < 5) {
            linear[(a0_b * 2 + a0_u * 2 + 1) % 5 * 72 + (a1_b * 2 + a1_u * 2 + 1)] = acc0__c1_1;
        }
    }
}
