// kernels for forward/backwarding in sigmiod layers

__kernel void sigmoid_forward(__constant float* prev_a,
        __read_only image2d_t weight,
        __constant float* bias,
        __global float* cur_z,
        __global float* cur_a,
        __constant float* dropout_coeffs,
        const int prev_d,
        const int cur_d)
{
    int idxc = get_global_id(0);

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

    // matrix-vector multiplication
    cur_z[idxc] = 0;
    for (int idxp = 0; idxp < prev_d; idxp++)
    {
        float4 w_val = read_imagef(weight, sampler,
                (int2)(idxp, idxc));
        cur_z[idxc] += w_val.x * prev_a[idxp];
    }
    cur_z[idxc] += bias[idxc];

    float tmp_a = 1.0 / (1.0 + exp(-cur_z[idxc]));
    cur_a[idxc] = tmp_a * dropout_coeffs[idxc];
}

__kernel void sigmoid_forward_img(__read_only image3d_t prev_a,
        __read_only image2d_t weight,
        __constant float* bias,
        __global float* cur_z,
        __global float* cur_a,
        __constant float* dropout_coeffs,
        const int prev_d,
        const int cur_d)
{
    int idxc = get_global_id(0);
    const int4 pa_dim = get_image_dim(prev_a);

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

    // matrix-vector multiplication
    cur_z[idxc] = 0;
    for (int idxp = 0; idxp < prev_d; idxp++)
    {
        float4 pa_val = read_imagef(prev_a, sampler,
                (int4)(idxp % pa_dim.x, (idxp / pa_dim.x) % pa_dim.y,
                    (idxp / pa_dim.x) / pa_dim.y, 0));
        float4 w_val = read_imagef(weight, sampler,
                (int2)(idxp, idxc));
        cur_z[idxc] += w_val.x * pa_val.x;
    }
    cur_z[idxc] += bias[idxc];

    float tmp_a = 1.0 / (1.0 + exp(-cur_z[idxc]));
    cur_a[idxc] = tmp_a * dropout_coeffs[idxc];
}
