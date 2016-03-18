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
    int g_idx = get_global_id(0);

    int delta_w = pool_width - (stride - 1);
    int delta_h = pool_height - (stride - 1);
    int out_width = (in_width - pool_width) / delta_w + 1;
    int out_height = (in_height - pool_height) / delta_h + 1;

    int out_x = g_idx % out_width;
    int out_y = (g_idx / out_width) % out_height;
    int out_m = (g_idx / (out_width * out_height)) % map_num;
    int out_t = (g_idx / (out_width * out_height * map_num));
    int map_off = (out_t * map_num + out_m) * in_width * in_height;

    // downsample cur_z
    float maxv = prev_z[((out_y * delta_h) * in_width + (out_x * delta_w)) + map_off];
    for (int y = out_y * delta_h; y < out_y * delta_h + pool_height; y++)
    {
        for (int x = out_x * delta_w; x < out_x * delta_w + pool_width; x++)
        {
            int in_idx = (y * in_width + x) + map_off;
            if (maxv < prev_z[in_idx])
            {
                maxv = prev_z[in_idx];
            }
        }
    }
    cur_z[g_idx] = maxv;

    // downsample cur_a
    maxv = prev_a[((out_y * delta_h) * in_width + (out_x * delta_w)) + map_off];
    for (int y = out_y * delta_h; y < out_y * delta_h + pool_height; y++)
    {
        for (int x = out_x * delta_w; x < out_x * delta_w + pool_width; x++)
        {
            int in_idx = (y * in_width + x) + map_off;
            if (maxv < prev_a[in_idx])
            {
                maxv = prev_a[in_idx];
            }
        }
    }
    cur_a[g_idx] = maxv;
}
