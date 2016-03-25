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
        __global float* patch_buf,
        const int patch_w,
        const int patch_h,
        const int patch_x,
        const int patch_y,
        const int gap)
{
    const int gid = get_global_id(0);

    const int lx = gid % patch_w;
    const int ly = (gid / patch_w) % patch_h;
    const int lc = (gid / patch_w) / patch_h;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

    float4 val = read_imagef(img_in, sampler,
            (int2)(patch_x * gap + lx, patch_y * gap + ly));
    float4 coeff = {lc == 0, lc == 1, lc == 2, lc == 3};
    patch_buf[gid] = dot(val, coeff);
}
