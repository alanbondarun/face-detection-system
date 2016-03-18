// kernels for forward/backwarding in max pool layers

__kernel void max_pool_forward(__constant float* prev_z,
        __constant float* prev_a,
        __global float* cur_z,
        __global float* cur_a,
        const int map_num,
        const int in_width,
        const int in_height,
        const int pool_width,
        const int pool_height,
        const int stride,
        const int train_num)
{
    // TODO
}
