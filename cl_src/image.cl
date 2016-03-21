// image modification kernels

__kernel void grayscale_img(__read_only image2d_t img_in,
        __write_only image2d_t img_out)
{
    const int2 pos = {get_global_id(0), get_global_id(1)};

    __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

    float4 val = read_imagef(img_in, sampler, pos);
    const float4 coeff = {0.2125, 0.7154, 0.0721, 1};
    val *= coeff;
    float vsum = val.x + val.y + val.z;
    write_imagef(img_out, pos, (float4)(vsum, vsum, vsum, val.w));
}
