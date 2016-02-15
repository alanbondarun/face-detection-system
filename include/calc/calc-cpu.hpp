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
	
	/* apply function to a vector */
	void apply_vec(double *v, size_t dim, double(*func)(const double));
	void apply_vec(double *v, size_t dim, std::function<double(double)> func);
	
	/* transpose a given matrix */
	void transpose_mat(const double *m, double *mres, size_t dim_r, size_t dim_c);
	
	/* sum up a set of vectors */
	void sum_vec(const double *vset, double *vres, size_t dim_v, size_t num_v);

	/* vector outer product */
	void vec_outer_prod(const double *v1, const double *v2, double *mres, size_t dim_n, size_t dim_m);
	
	/* calculate downsampling */
	void downsample_max(const double *m, const double *mres, size_t dim_w, size_t dim_h, size_t pool_w, size_t pool_h);
	
	/* calculate upsampling */
	void upsample_max(const double *me, const double *ma, double *me_res,
			size_t dim_w, size_t dim_h, size_t pool_w, size_t pool_h);
}

#endif // __CALC_CPU_HPP