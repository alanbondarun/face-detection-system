// kernels for forward/backwarding in sigmiod layers

__kernel void sigmoid_forward(__read_only image2d_t prev_a,
        __read_only image2d_t weight,
        __constant float* bias,
        __write_only image2d_t cur_z,
        __write_only image2d_t cur_a,
        __constant float* dropout_coeffs,
        const int prev_d,
        const int cur_d)
{
    int2 gpos = {get_global_id(0), get_global_id(1)};
    int idxc = gpos.x;
    int idxt = gpos.y;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

    // matrix-vector multiplication
    float tmpz = 0;
    for (int idxp = 0; idxp < prev_d; idxp++)
    {
        float4 w_val = read_imagef(weight, sampler,
                (int2)(idxp, idxc));
        float4 pa_val = read_imagef(prev_a, sampler,
                (int2)(idxp, idxt));
        tmpz += w_val.x * pa_val.x;
    }
    tmpz += bias[idxc];
    write_imagef(cur_z, gpos, tmpz);

    float tmpa = (1.0 / (1.0 + exp(-tmpz))) * dropout_coeffs[idxc];
    write_imagef(cur_a, gpos, tmpa);
}

__kernel void sigmoid_forward_img(__read_only image3d_t prev_a,
        __read_only image2d_t weight,
        __constant float* bias,
        __write_only image2d_t cur_z,
        __write_only image2d_t cur_a,
        __constant float* dropout_coeffs,
        const int prev_d,
        const int cur_d)
{
    int2 gpos = {get_global_id(0), get_global_id(1)};
    int idxc = gpos.x;
    int idxt = gpos.y;
    const int4 pa_dim = get_image_dim(prev_a);

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

    // matrix-vector multiplication
    float tmpz = 0;
    for (int idxp = 0; idxp < prev_d; idxp++)
    {
        float4 pa_val = read_imagef(prev_a, sampler,
                (int4)((idxp % pa_dim.x), (idxp / pa_dim.x) % pa_dim.y,
                    idxt, 0));
        float4 w_val = read_imagef(weight, sampler,
                (int2)(idxp, idxc));
        tmpz += w_val.x * pa_val.x;
    }
    tmpz += bias[idxc];
    write_imagef(cur_z, gpos, (float4)(tmpz));

    float tmpa = (1.0 / (1.0 + exp(-tmpz))) * dropout_coeffs[idxc];
    write_imagef(cur_a, gpos, (float4)(tmpa));
}
