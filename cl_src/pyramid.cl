// kernels for pyramid imaging

__kernel void shrink_image(__read_only image2d_t img_in,
        __write_only image2d_t img_out)
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int2 in_dim = get_image_dim(img_in);
    const int2 out_dim = get_image_dim(img_out);

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

    float posx = (float)(pos.x * in_dim.x) / out_dim.x;
    float posy = (float)(pos.y * in_dim.y) / out_dim.y;
    int iposx = (int)(posx);
    int iposy = (int)(posy);
    float dposx = posx - (float)(iposx);
    float dposy = posy - (float)(iposy);

    float4 val = 0;
    val += (1-dposx) * (1-dposy) * (read_imagef(img_in, sampler, (int2)(iposx, iposy)));
    val += (dposx) * (1-dposy) * (read_imagef(img_in, sampler, (int2)(iposx+1, iposy)));
    val += (1-dposx) * (dposy) * (read_imagef(img_in, sampler, (int2)(iposx, iposy+1)));
    val += (dposx) * (dposy) * (read_imagef(img_in, sampler, (int2)(iposx+1, iposy+1)));

    write_imagef(img_out, pos, val);
}

__kernel void extract_image_patches(__read_only image2d_t img_in,
        __global float* patch_array,
        const int patch_width,
        const int patch_height,
        const int gap,
        const int channel_width)
{
    const int idx = get_global_id(0);
    const int2 in_dim = get_image_dim(img_in);

    const int rown = (in_dim.x - patch_width) / gap + 1;

    const int gn = idx / (patch_width * patch_height * channel_width);
    const int gx = gn % rown;
    const int gy = gn / rown;

    const int lnn = (idx / channel_width) % (patch_width * patch_height);
    const int lx = lnn % patch_width;
    const int ly = lnn / patch_width;

    const int ic = idx % channel_width;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

    float4 val = read_imagef(img_in, sampler, (int2)(gx * gap + lx, gy * gap + ly));
    float4 coeff = {ic == 0, ic == 1, ic == 2, ic == 3};
    patch_array[idx] = dot(val, coeff);
}
