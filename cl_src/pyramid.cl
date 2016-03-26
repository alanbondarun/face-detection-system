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
        __write_only image3d_t patch_imgbuf,
        const int patch_w,
        const int gap)
{
    const int4 gpos = {get_global_id(0), get_global_id(1), get_global_id(2),
        0};

    const int pos_x = gpos.x % patch_w;
    const int pos_y = gpos.x / patch_w;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

    float4 val = read_imagef(img_in, sampler,
            (int2)(gpos.y * gap + pos_x, gpos.z * gap + pos_y));
    write_imagef(patch_imgbuf, gpos, val);
}

__kernel void patch_create_cdf(__read_only image3d_t patch_imgbuf,
        __write_only image3d_t cdf_buf)
{
    const int2 patch_pos = {get_global_id(0), get_global_id(1)};
    const int psize = get_image_dim(patch_imgbuf).x;
    const int level = get_image_dim(cdf_buf).x;

    __private int cdf_val[256];
    for (int i=0; i<level; i++)
        cdf_val[i] = 0;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

    for (int i=0; i<psize; i++)
    {
        int idx = floor(read_imagef(patch_imgbuf, sampler,
                (int4)(i, patch_pos.x, patch_pos.y, 0)).x * level);
        cdf_val[clamp(idx, 0, level-1)]++;
    }

    write_imagef(cdf_buf, (int4)(0, patch_pos.x, patch_pos.y, 0),
            (float4)(cdf_val[0]));
    for (int i=1; i<level; i++)
    {
        cdf_val[i] += cdf_val[i-1];
        write_imagef(cdf_buf, (int4)(i, patch_pos.x, patch_pos.y, 0),
                (float4)(cdf_val[i]));
    }
}

__kernel void patch_apply_cdf(__read_only image3d_t patch_in,
        __read_only image3d_t patch_cdf,
        __write_only image3d_t patch_out)
{
    const int4 gpos = {get_global_id(0), get_global_id(1), get_global_id(2),
        0};
    const int psize = get_image_dim(patch_in).x;
    const int level = get_image_dim(patch_cdf).x;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

    float4 in_val = read_imagef(patch_in, sampler, gpos);
    int in_idx = floor(read_imagef(patch_in, sampler, gpos).x * level);
    float4 out_val = read_imagef(patch_cdf, sampler,
            (int4)(clamp(in_idx, 0, level-1), gpos.y, gpos.z, 0)) / psize;

    write_imagef(patch_out, gpos, out_val);
}

__kernel void patch_get_lsq_coeff(__read_only image3d_t patch_in,
        __write_only image3d_t coeff_buf,
        const int patch_w)
{
    const int2 patch_pos = {get_global_id(0), get_global_id(1)};
    const int psize = get_image_dim(patch_in).x;
    const int patch_h = psize / patch_w;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

    float xsum = 0, ysum = 0, csum = 0;
    for (int i=0; i<psize; i++)
    {
        float tmpv = read_imagef(patch_in, sampler,
                (int4)(i, patch_pos.x, patch_pos.y, 0)).x;
        xsum += tmpv * (i % patch_w);
        ysum += tmpv * (i / patch_w);
        csum += tmpv;
    }

    float x2s = (patch_h * (patch_w*(patch_w-1)*(2*patch_w-1))) / 6.0;
    float xys = (patch_w * (patch_w-1) * patch_h * (patch_h-1)) / 4.0;
    float xs = (patch_h * patch_w * (patch_w-1)) / 2.0;
    float y2s = (patch_w * (patch_h*(patch_h-1)*(2*patch_h-1))) / 6.0;
    float ys = (patch_w * patch_h * (patch_h-1)) / 2.0;
    float cs = patch_w * patch_h;

    float16 cmat;
    cmat.s0 = y2s * cs - ys * ys;
    cmat.s1 = -(xys * cs - ys * xs);
    cmat.s2 = xys * ys - xs * y2s;
    cmat.s3 = cmat.s1;
    cmat.s4 = x2s * cs - xs * xs;
    cmat.s5 = -(x2s * ys - xys * xs);
    cmat.s6 = cmat.s2;
    cmat.s7 = cmat.s5;
    cmat.s8 = x2s * y2s - xys * xys;

    float det = x2s * cmat.s0 + xys * cmat.s1 + xs * cmat.s2;
    cmat /= det;

    float coeff1 = cmat.s0 * xsum + cmat.s1 * ysum + cmat.s2 * csum;
    float coeff2 = cmat.s3 * xsum + cmat.s4 * ysum + cmat.s5 * csum;
    float coeff3 = cmat.s6 * xsum + cmat.s5 * ysum + cmat.s8 * csum;

    write_imagef(coeff_buf, (int4)(0, patch_pos.x, patch_pos.y, 0),
            (float4)(coeff1));
    write_imagef(coeff_buf, (int4)(1, patch_pos.x, patch_pos.y, 0),
            (float4)(coeff2));
    write_imagef(coeff_buf, (int4)(2, patch_pos.x, patch_pos.y, 0),
            (float4)(coeff3));
}

__kernel void patch_apply_lsq(__read_only image3d_t patch_in,
        __read_only image3d_t coeff_buf,
        __write_only image3d_t patch_out,
        const int patch_w)
{
    const int4 gpos = {get_global_id(0), get_global_id(1), get_global_id(2),
        0};
    const int gx = gpos.x % patch_w;
    const int gy = gpos.x / patch_w;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

    float4 in_val = read_imagef(patch_in, sampler, gpos);
    float4 coeff_x = read_imagef(coeff_buf, sampler,
            (int4)(0, gpos.y, gpos.z, 0));
    float4 coeff_y = read_imagef(coeff_buf, sampler,
            (int4)(1, gpos.y, gpos.z, 0));
    float4 coeff_c = read_imagef(coeff_buf, sampler,
            (int4)(2, gpos.y, gpos.z, 0));

    float4 out_val = in_val - gx * coeff_x - gy * coeff_y - coeff_c;
    write_imagef(patch_out, gpos, out_val);
}

__kernel void patch_get_mean_var(__read_only image3d_t patch_in,
        __write_only image3d_t meanvar_buf)
{
    const int2 patch_pos = {get_global_id(0), get_global_id(1)};
    const int psize = get_image_dim(patch_in).x;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

    float cmean = 0;
    for (int i=0; i<psize; i++)
    {
        cmean += read_imagef(patch_in, sampler,
                (int4)(i, patch_pos.x, patch_pos.y, 0)).x;
    }
    cmean /= psize;

    float cvar = 0;
    for (int i=0; i<psize; i++)
    {
        cvar += pown(read_imagef(patch_in, sampler,
                (int4)(i, patch_pos.x, patch_pos.y, 0)).x - cmean, 2);
    }
    cvar /= psize;

    write_imagef(meanvar_buf, (int4)(0, patch_pos.x, patch_pos.y, 0),
            (float4)(cmean));
    write_imagef(meanvar_buf, (int4)(1, patch_pos.x, patch_pos.y, 0),
            (float4)(cvar));
}

__kernel void patch_normalize(__read_only image3d_t patch_in,
        __read_only image3d_t meanvar_buf,
        __write_only image3d_t patch_out,
        const int patch_w,
        const float new_mean,
        const float new_stdev)
{
    const int4 gpos = {get_global_id(0), get_global_id(1), get_global_id(2),
        0};
    const int gx = gpos.x % patch_w;
    const int gy = gpos.x / patch_w;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

    float4 in_val = read_imagef(patch_in, sampler, gpos);
    float4 cmean = read_imagef(meanvar_buf, sampler,
            (int4)(0, gpos.y, gpos.z, 0));
    float4 cvar = read_imagef(meanvar_buf, sampler,
            (int4)(1, gpos.y, gpos.z, 0));

    float var1 = isnormal(cvar.x);
    float var0 = 1.0 - var1;
    float alpha = new_stdev * sqrt(1.0 / cvar.x);

    float4 out_val = var1 * (alpha * (in_val - cmean) + new_mean)
        + var0 * new_mean;
    write_imagef(patch_out, gpos, out_val);
}
