#ifndef __CALC_CPU_HPP
#define __CALC_CPU_HPP

#include <cstdlib>
#include <functional>

namespace NeuralNet
{
    /* addition of vectors */
    void add_vec(const float *v1, const float *v2, float *vres, size_t dim);

    /* pairwise multiplication of vectors */
    void pmul_vec(const float *v1, const float *v2, float *vres, size_t dim);

    /* vector-matrix multiplication: (r x c) matrix and c-ary vector */
    void mul_mat_vec(const float *m, const float *v, float *vres, size_t dim_r, size_t dim_c);

    /* copy a vector */
    void copy_vec(const float *v, float *vres, size_t dim);

    // set each element of the vector to the given value
    void set_vec(float *v, float val, size_t dim);

    // multiply each element of the vector to the given value
    void const_mul_vec(float *v, float val, size_t dim);

    /* apply function to a vector */
    void apply_vec(const float *v, float *vres, size_t dim, std::function<float(float)> func);

    /* transpose a given matrix */
    void transpose_mat(const float *m, float *mres, size_t dim_r, size_t dim_c);

    /* sum up a set of vectors */
    void sum_vec(const float *vset, float *vres, size_t dim_v, size_t num_v);

    /* vector outer product */
    void vec_outer_prod(const float *v1, const float *v2, float *mres, size_t dim_n, size_t dim_m);

    /* calculate downsampling */
    void downsample_max(const float *m, float *mres, size_t dim_w, size_t dim_h, size_t pool_w,
            size_t pool_h, size_t stride);

    /* calculate upsampling */
    void upsample_max(const float *me, const float *ma, float *me_res,
            size_t dim_w, size_t dim_h, size_t pool_w, size_t pool_h, size_t stride);

    /* flip (rotate 180 deg) a matrix */
    void flip_mat(const float *m, float *mres, size_t dim_w, size_t dim_h);

    void inflate_mats(const float *m_in, float *m_res, size_t dim_w, size_t dim_h,
            size_t pad, size_t num_m);

    struct MatrixRange
    {
        int x, y, w, h;
        MatrixRange(int _x, int _y, int _w, int _h)
            : x(_x), y(_y), w(_w), h(_h) {}
    };

    /* calculate convolution */
    void convolution_mat(const float *m_in, const float *m_conv, float *m_res,
            int dim_w, int dim_h, int dim_conv_w, int dim_conv_h,
            const MatrixRange& range);

    void convolution_mat_no_zeros(const float *m_in, const float *m_conv, float *m_res,
            int dim_w, int dim_h, int dim_conv_w, int dim_conv_h);
    void convolution_mat_same_zeros(const float *m_in, const float *m_conv, float *m_res,
            int dim_w, int dim_h, int dim_conv_w, int dim_conv_h);
    void convolution_mat_wide_zeros(const float *m_in, const float *m_conv, float *m_res,
            int dim_w, int dim_h, int dim_conv_w, int dim_conv_h);
}

#endif // __CALC_CPU_HPP
