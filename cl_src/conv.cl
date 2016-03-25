// kernels for forward/backwarding in convolution layers

__kernel void conv_forward_relu_zeropad(__read_only image3d_t prev_a,
        __write_only image3d_t cur_z,
        __write_only image3d_t cur_a,
        __read_only image3d_t weight,
        __read_only image3d_t bias)
{
    const int4 out_pos = {get_global_id(0), get_global_id(1),
        get_global_id(2), 0};
    const int4 in_dim = get_image_dim(prev_a);
    const int4 out_dim = get_image_dim(cur_a);
    const int4 w_dim = get_image_dim(weight);
    const int recep_size = (int)(sqrt((float)(w_dim.x)));
    const int recep_hw = (recep_size - 1) / 2;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

    float cz_val = 0;
    for (int idxpm = 0; idxpm < in_dim.z; idxpm++)
    {
        int min_cx = max(0, recep_hw - out_pos.x);
        int max_cx = min(recep_size, in_dim.x + recep_hw - out_pos.x);
        int min_cy = max(0, recep_hw - out_pos.y);
        int max_cy = min(recep_size, in_dim.y + recep_hw - out_pos.y);
        for (int conv_y = min_cy; conv_y < max_cy; conv_y++)
        {
            for (int conv_x = min_cx; conv_x < max_cx; conv_x++)
            {
                int prev_x = out_pos.x + conv_x - recep_hw;
                int prev_y = out_pos.y + conv_y - recep_hw;
                float4 pa_val = read_imagef(prev_a, sampler,
                        (int4)(prev_x, prev_y, idxpm, 0));
                float4 w_val = read_imagef(weight, sampler,
                        (int4)(recep_size - 1 - conv_x +
                            (recep_size - 1 - conv_y) * recep_size,
                            idxpm, out_pos.z, 0));
                cz_val += (pa_val.x * w_val.x);
            }
        }
    }

    float4 bias_val = read_imagef(bias, sampler,
            (int4)(out_pos.x, out_pos.y, out_pos.z, 0));
    cz_val += bias_val.x;
    write_imagef(cur_z, out_pos, (float4)(cz_val));

    float tmp_z = fabs(cz_val);
    write_imagef(cur_a, out_pos, (float4)((cz_val + tmp_z) / 2.0));
}
