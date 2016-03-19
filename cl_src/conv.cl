// kernels for forward/backwarding in convolution layers

__kernel void conv_forward_relu_zeropad(__constant float* prev_a,
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
    int g_idx = get_global_id(0);

    int out_x = g_idx % in_width;
    int out_y = (g_idx / in_width) % in_height;
    int out_m = (g_idx / (in_width * in_height)) % out_maps;
    int out_t = (g_idx / (in_width * in_height * out_maps));
    int recep_hw = (recep_size - 1) / 2;

    cur_z[g_idx] = 0;
    int pa_off = out_t * in_maps * in_width * in_height;
    int map_off = out_m * in_maps * recep_size * recep_size;
    for (int idxpm = 0; idxpm < in_maps; idxpm++)
    {
        for (int conv_y = 0; conv_y < recep_size; conv_y++)
        {
            for (int conv_x = 0; conv_x < recep_size; conv_x++)
            {
                int prev_x = out_x + conv_x - recep_hw;
                int prev_y = out_y + conv_y - recep_hw;
                float enable_coeff = (float)((prev_x >= 0) && (prev_y >= 0) &&
                        (prev_x < in_width) && (prev_y < in_height));
                cur_z[g_idx] += (enable_coeff *
                        prev_a[pa_off + prev_y * in_width + prev_x] *
                        weight[map_off + (recep_size - 1 - conv_y) * recep_size +
                            (recep_size - 1 - conv_x)]);
            }
        }

        pa_off += (in_width * in_height);
        map_off += (recep_size * recep_size);
    }

    cur_z[g_idx] += bias[g_idx];
    
    float tmp_z = fabs(cur_z[g_idx]);
    cur_a[g_idx] = (cur_z[g_idx] + tmp_z) / 2.0;
}
