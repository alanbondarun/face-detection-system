// kernels for forward/backwarding in sigmiod layers

__kernel void sigmoid_forward(__constant float* prev_a,
        __constant float* weight,
        __constant float* bias,
        __global float* cur_z,
        __global float* cur_a,
        __global float* dropout_coeffs,
        const int prev_d,
        const int cur_d,
        const int train_num)
{
    // TODO
}
