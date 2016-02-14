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
}

#endif // __CALC_CPU_HPP