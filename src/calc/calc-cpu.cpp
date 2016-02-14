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
	
	void apply_vec(double *v, size_t dim, std::function<double(double)> func)
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
	
	void sum_vec(const double *vset, double *vres, size_t dim_v, size_t num_v)
	{
		double *p_vres = vres;
		const double *p_vset = vset;
		
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
	
	void vec_outer_prod(const double *v1, const double *v2, double *mres, size_t dim_n, size_t dim_m)
	{
		const double *p_v1 = v1;
		const double *p_v2 = v2;
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
}