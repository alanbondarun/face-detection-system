// kernels for forward/backwarding in max pool layers

__kernel void max_pool_forward(__read_only image3d_t prev_z,
        __read_only image3d_t prev_a,
        __write_only image3d_t cur_z,
        __write_only image3d_t cur_a,
        const int pool_width,
        const int pool_height,
        const int stride)
{
    const int4 out_pos = {get_global_id(0), get_global_id(1),
        get_global_id(2), 0};
    const int4 in_dim = get_image_dim(prev_z);

    int delta_w = pool_width - (stride - 1);
    int delta_h = pool_height - (stride - 1);

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

    // downsample cur_z
    float maxv = read_imagef(prev_z, sampler,
            (int4)(out_pos.x * delta_w, out_pos.y * delta_h, out_pos.z, 0)).x;
    for (int y = out_pos.y * delta_h; y < out_pos.y * delta_h + pool_height;
            y++)
    {
        for (int x = out_pos.x * delta_w; x < out_pos.x * delta_w + pool_width;
                x++)
        {
            maxv = fmax(maxv,
                    read_imagef(prev_z, sampler,
                        (int4)(x, y, out_pos.z, 0)).x);
        }
    }
    write_imagef(cur_z, out_pos, (float4)(maxv, maxv, maxv, maxv));

    // downsample cur_a
    maxv = read_imagef(prev_a, sampler,
            (int4)(out_pos.x * delta_w, out_pos.y * delta_h, out_pos.z, 0)).x;
    for (int y = out_pos.y * delta_h; y < out_pos.y * delta_h + pool_height;
            y++)
    {
        for (int x = out_pos.x * delta_w; x < out_pos.x * delta_w + pool_width;
                x++)
        {
            maxv = fmax(maxv,
                    read_imagef(prev_a, sampler,
                        (int4)(x, y, out_pos.z, 0)).x);
        }
    }
    write_imagef(cur_a, out_pos, (float4)(maxv, maxv, maxv, maxv));
}
