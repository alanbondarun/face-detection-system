#include "calc/calc-cpu.hpp"

namespace NeuralNet
{
    void add_vec(const float *v1, const float *v2, float *vres, size_t dim)
    {
        float *pres = vres;
        while (vres - pres < dim)
        {
            *vres = *v1 + *v2;
            vres++;
            v1++;
            v2++;
        }
    }

    void pmul_vec(const float *v1, const float *v2, float *vres, size_t dim)
    {
        float *pres = vres;
        while (vres - pres < dim)
        {
            *vres = *v1 * *v2;
            vres++;
            v1++;
            v2++;
        }
    }

    void mul_mat_vec(const float *m, const float *v, float *vres, size_t dim_r, size_t dim_c)
    {
        float *pres = vres;
        const float *pv = v;
        while (vres - pres < dim_r)
        {
            *vres = 0;
            while (v - pv < dim_c)
            {
                *vres += (*m) * (*v);
                m++;
                v++;
            }
            vres++;
            v = pv;
        }
    }

    void copy_vec(const float *v, float *vres, size_t dim)
    {
        const float *pres = v;
        while (v - pres < dim)
        {
            *vres = *v;
            v++;
            vres++;
        }
    }

    void set_vec(float *v, float val, size_t dim)
    {
        float *pv = v;
        while (v - pv < dim)
        {
            *v = val;
            v++;
        }
    }

    void const_mul_vec(float *v, float val, size_t dim)
    {
        float *pv = v;
        while (v - pv < dim)
        {
            *v *= val;
            v++;
        }
    }

    void apply_vec(const float *v, float *vres, size_t dim, std::function<float(float)> func)
    {
        const float *pres = v;
        while (v - pres < dim)
        {
            *vres = func(*v);
            v++;
            vres++;
        }
    }

    void transpose_mat(const float *m, float *mres, size_t dim_r, size_t dim_c)
    {
        for (int i=0; i<dim_r; i++)
        {
            for (int j=0; j<dim_c; j++)
            {
                mres[i+j*dim_r] = m[i*dim_c+j];
            }
        }
    }

    void sum_vec(const float *vset, float *vres, size_t dim_v, size_t num_v)
    {
        float *p_vres = vres;
        const float *p_vset = vset;

        /* initialize to zero before addition */
        while (vres - p_vres < dim_v)
        {
            *vres = 0;
            vres++;
        }
        vres = p_vres;

        while (vset - p_vset < dim_v*num_v)
        {
            while (vres - p_vres < dim_v)
            {
                *vres += *vset;
                vres++;
                vset++;
            }
            vres = p_vres;
        }
    }

    void vec_outer_prod(const float *v1, const float *v2, float *mres, size_t dim_n, size_t dim_m)
    {
        const float *p_v1 = v1;
        const float *p_v2 = v2;
        while (v1 - p_v1 < dim_n)
        {
            while (v2 - p_v2 < dim_m)
            {
                *mres = (*v1) * (*v2);
                mres++;
                v2++;
            }
            v1++;
            v2 = p_v2;
        }
    }

    void downsample_max(const float *m, float *mres, size_t dim_w, size_t dim_h, size_t pool_w,
            size_t pool_h, size_t stride)
    {
        const size_t delta_w = pool_w - (stride - 1);
        const size_t delta_h = pool_h - (stride - 1);
        const size_t ratio_w = (dim_w - pool_w) / delta_w + 1;
        for (size_t i=0; i + pool_h <= dim_h; i += delta_h)
        {
            for (size_t j=0; j + pool_w <= dim_w; j += delta_w)
            {
                float vmax = m[i*dim_w + j];
                for (size_t y=i; y < i+pool_h; y++)
                {
                    for (size_t x=j; x < j+pool_w; x++)
                    {
                        vmax = std::max(vmax, m[y*dim_w + x]);
                    }
                }
                mres[(i/delta_h) * ratio_w + (j/delta_w)] = vmax;
            }
        }
    }

    void upsample_max(const float *me, const float *ma, float *me_res,
            size_t dim_w, size_t dim_h, size_t pool_w, size_t pool_h, size_t stride)
    {
        const size_t delta_w = pool_w - (stride - 1);
        const size_t delta_h = pool_h - (stride - 1);
        const size_t ratio_w = (dim_w - pool_w) / delta_w + 1;

        for (size_t i=0; i<dim_w * dim_h; i++)
        {
            me_res[i] = 0;
        }

        for (size_t i=0; i + pool_h <= dim_h; i += delta_h)
        {
            for (size_t j=0; j + pool_w <= dim_w; j += delta_w)
            {
                auto max_x = j, max_y = i;
                float vmax = ma[i*dim_w + j];
                for (size_t y=i; y < i+pool_h; y++)
                {
                    for (size_t x=j; x < j+pool_w; x++)
                    {
                        if (vmax < ma[y*dim_w + x])
                        {
                            vmax = ma[y*dim_w + x];
                            max_x = x;
                            max_y = y;
                        }
                    }
                }
                me_res[max_x * dim_w + max_y] = me[(i/delta_h) * ratio_w + (j/delta_w)];
            }
        }
    }

    void flip_mat(const float *m, float *mres, size_t dim_w, size_t dim_h)
    {
        const float *pm = m;
        float *pmres = mres + (dim_w*dim_h - 1);

        if (m == mres)
        {
            /* TODO */
        }
        else
        {
            while (m - pm < dim_w * dim_h)
            {
                *pmres = *m;
                m++;
                pmres--;
            }
        }
    }

    void inflate_mats(const float *m_in, float *m_res, size_t dim_w, size_t dim_h,
            size_t pad, size_t num_m)
    {
        const size_t out_w = dim_w + pad*2;
        const size_t out_h = dim_h + pad*2;

        for (size_t i=0; i<num_m; i++)
        {
            size_t in_off = dim_w * dim_h * i;
            size_t res_off = out_w * out_h * i;

            for (size_t y = 0; y < out_h; y++)
            {
                for (size_t x = 0; x < out_w; x++)
                {
                    if (x>=pad && x<pad+dim_w && y>=pad && y<pad+dim_h)
                    {
                        m_res[y*out_w + x + res_off] = m_in[(y-pad)*dim_w + (x-pad) + in_off];
                    }
                    else
                    {
                        m_res[y*out_w + x + res_off] = 0;
                    }
                }
            }
        }
    }

    void convolution_mat(const float *m_in, const float *m_conv, float *m_res,
            int dim_w, int dim_h, int dim_conv_w, int dim_conv_h,
            const MatrixRange& range)
    {
        const int i_dim_w = dim_w;
        const int i_dim_h = dim_h;
        for (int j=0;j<range.h;j++) {
            for (int i=0;i<range.w;i++) {
                float sum=0;
                for (int jj=0;jj<dim_conv_h;jj++) {
                    for (int ii=0;ii<dim_conv_w;ii++) {
                        int in_i = range.x + i + dim_conv_w - ii - 1;
                        int in_j = range.y + j + dim_conv_h - jj - 1;
                        if (in_i >= 0 && in_i < i_dim_w && in_j >= 0 && in_j < i_dim_h)
                            sum += m_conv[jj*dim_conv_w + ii] * m_in[in_j*dim_w + in_i];
                    }
                }
                m_res[j*range.w + i] = sum;
            }
        }
    }

    void convolution_mat_no_zeros(const float *m_in, const float *m_conv, float *m_res,
            int dim_w, int dim_h, int dim_conv_w, int dim_conv_h)
    {
        convolution_mat(m_in, m_conv, m_res, dim_w, dim_h, dim_conv_w, dim_conv_h, MatrixRange(
            0, 0, dim_w - dim_conv_w + 1, dim_h - dim_conv_h + 1
        ));
    }

    void convolution_mat_same_zeros(const float *m_in, const float *m_conv, float *m_res,
            int dim_w, int dim_h, int dim_conv_w, int dim_conv_h)
    {
        convolution_mat(m_in, m_conv, m_res, dim_w, dim_h, dim_conv_w, dim_conv_h, MatrixRange(
            -(dim_conv_w/2), -(dim_conv_h/2), dim_w, dim_h
        ));
    }

    void convolution_mat_wide_zeros(const float *m_in, const float *m_conv, float *m_res,
            int dim_w, int dim_h, int dim_conv_w, int dim_conv_h)
    {
        convolution_mat(m_in, m_conv, m_res, dim_w, dim_h, dim_conv_w, dim_conv_h, MatrixRange(
            -dim_conv_w+1, -dim_conv_h+1, dim_w+dim_conv_w-1, dim_h+dim_conv_h-1
        ));
    }
}
