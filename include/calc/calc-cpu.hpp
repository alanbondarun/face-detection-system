#ifndef __CALC_CPU_HPP
#define __CALC_CPU_HPP

#include <cstdlib>
#include <functional>

namespace NeuralNet
{
    /* addition of vectors */
    void add_vec(const double *v1, const double *v2, double *vres, size_t dim);

    /* pairwise multiplication of vectors */
    void pmul_vec(const double *v1, const double *v2, double *vres, size_t dim);

    /* vector-matrix multiplication: (r x c) matrix and c-ary vector */
    void mul_mat_vec(const double *m, const double *v, double *vres, size_t dim_r, size_t dim_c);

    /* copy a vector */
    void copy_vec(const double *v, double *vres, size_t dim);

    // set each element of the vector to the given value
    void set_vec(double *v, double val, size_t dim);

    // multiply each element of the vector to the given value
    void const_mul_vec(double *v, double val, size_t dim);

    /* apply function to a vector */
    void apply_vec(const double *v, double *vres, size_t dim, std::function<double(double)> func);

    /* transpose a given matrix */
    void transpose_mat(const double *m, double *mres, size_t dim_r, size_t dim_c);

    /* sum up a set of vectors */
    void sum_vec(const double *vset, double *vres, size_t dim_v, size_t num_v);

    /* vector outer product */
    void vec_outer_prod(const double *v1, const double *v2, double *mres, size_t dim_n, size_t dim_m);

    /* calculate downsampling */
    void downsample_max(const double *m, double *mres, size_t dim_w, size_t dim_h, size_t pool_w,
            size_t pool_h, size_t stride);

    /* calculate upsampling */
    void upsample_max(const double *me, const double *ma, double *me_res,
            size_t dim_w, size_t dim_h, size_t pool_w, size_t pool_h, size_t stride);

    /* flip (rotate 180 deg) a matrix */
    void flip_mat(const double *m, double *mres, size_t dim_w, size_t dim_h);

    struct MatrixRange
    {
        int x, y, w, h;
        MatrixRange(int _x, int _y, int _w, int _h)
            : x(_x), y(_y), w(_w), h(_h) {}
    };

    /* calculate convolution */
    void convolution_mat(const double *m_in, const double *m_conv, double *m_res,
            int dim_w, int dim_h, int dim_conv_w, int dim_conv_h,
            const MatrixRange& range);

    void convolution_mat_no_zeros(const double *m_in, const double *m_conv, double *m_res,
            int dim_w, int dim_h, int dim_conv_w, int dim_conv_h);
    void convolution_mat_same_zeros(const double *m_in, const double *m_conv, double *m_res,
            int dim_w, int dim_h, int dim_conv_w, int dim_conv_h);
    void convolution_mat_wide_zeros(const double *m_in, const double *m_conv, double *m_res,
            int dim_w, int dim_h, int dim_conv_w, int dim_conv_h);

    // calculate local-response normalizations for convolution neurons
    void lr_normalize_mat(const double *m, double *m_res, size_t dim_w, size_t dim_h,
            size_t reg_size, double alpha, double beta);

    // calculate normalization term of error propagation
    // for local-response normalization layers
    void lr_normalize_prime(const double *m, double *m_res, size_t dim_w, size_t dim_h,
            size_t reg_size, double alpha, double beta);
}

#endif // __CALC_CPU_HPP
