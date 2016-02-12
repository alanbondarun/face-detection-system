#include "calc/calc-cpu.hpp"

namespace NeuralNet
{
	void add_vec(const double *v1, const double *v2, double *vres, size_t dim)
	{
		double *pres = vres;
		while (vres - pres < dim)
		{
			*vres = *v1 + *v2;
			vres++;
			v1++;
			v2++;
		}
	}
	
	void pmul_vec(const double *v1, const double *v2, double *vres, size_t dim)
	{
		double *pres = vres;
		while (vres - pres < dim)
		{
			*vres = *v1 * *v2;
			vres++;
			v1++;
			v2++;
		}
	}
	
	void mul_mat_vec(const double *m, const double *v, double *vres, size_t dim_r, size_t dim_c)
	{
		double *pres = vres;
		const double *pv = v;
		while (vres - pres < dim_r)
		{
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
	
	void apply_vec(double *v, size_t dim, double(*func)(const double))
	{
		double *pres = v;
		while (v - pres < dim)
		{
			*v = func(*v);
			v++;
		}
	}
	
	void transpose_mat(const double *m, double *mres, size_t dim_r, size_t dim_c)
	{
		for (int i=0; i<dim_r; i++)
		{
			for (int j=0; j<dim_c; j++)
			{
				mres[i+j*dim_r] = m[i*dim_c+j];
			}
		}
	}
}