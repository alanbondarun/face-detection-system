// kernels for forward/backwarding in convolution layers

__kernel void conv_forward_relu(__constant float* prev_a,
        __global float* cur_z,
        __global float* cur_a,
        __constant float* weight,
        __constant float* bias,
        const int in_width,
        const int in_height,
        const int in_maps,
        const int out_maps,
        const int recep_size,
        const int train_num)
{
    // TODO
}
