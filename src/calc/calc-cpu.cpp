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
	
	void apply_vec(double *v, double *vres, size_t dim, double(*func)(const double))
	{
		double *pres = v;
		while (v - pres < dim)
		{
			*vres = func(*v);
			v++;
		}
	}
	
	void apply_vec(double *v, double *vres, size_t dim, std::function<double(double)> func)
	{
		double *pres = v;
		while (v - pres < dim)
		{
			*vres = func(*v);
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
	
	void downsample_max(const double *m, double *mres, size_t dim_w, size_t dim_h, size_t pool_w, size_t pool_h)
	{
		const size_t ratio_w = dim_w / pool_w;
		double vmax;
		for (int i=0; i<dim_h; i += pool_h)
		{
			for (int j=0; j<dim_w; j += pool_w)
			{
				vmax = m[i*dim_w + j];
				for (int y=i; y<std::min(i+pool_h, dim_h); y++)
				{
					for (int x=j; x<std::min(j+pool_w, dim_w); x++)
					{
						vmax = std::max(vmax, m[y*dim_w + x]);
					}
				}
				mres[(i/pool_h) * ratio_w + (j/pool_w)] = vmax;
			}
		}
	}
	
	void upsample_max(const double *me, const double *ma, double *me_res,
			size_t dim_w, size_t dim_h, size_t pool_w, size_t pool_h)
	{
		const size_t ratio_w = dim_w / pool_w;
		const size_t ratio_h = dim_h / pool_h;
		
		for (int i=0; i<dim_w * dim_h; i++)
		{
			me_res[i] = 0;
		}
		
		for (int i=0; i<dim_h; i += pool_h)
		{
			for (int j=0; j<dim_w; j += pool_w)
			{
				int max_x = 0, max_y = 0;
				double vmax = ma[i*dim_w + j];
				for (int y=i; y<std::min(i+pool_h, dim_h); y++)
				{
					for (int x=j; x<std::min(j+pool_w, dim_w); x++)
					{
						if (vmax < ma[y*dim_w + x])
						{
							vmax = ma[y*dim_w + x];
							max_x = x;
							max_y = y;
						}
					}
				}
				me_res[max_x * dim_w + max_y] = me[(i/pool_h) * ratio_w + (j/pool_w)];
			}
		}
	}
	
	void flip_mat(const double *m, double *mres, size_t dim_w, size_t dim_h)
	{
		const double *pm = m;
		double *pmres = mres + (dim_w*dim_h - 1);
		
		while (m - pm < dim_w * dim_h)
		{
			*mres = *m;
			m++;
			mres--;
		}
	}
	
	void convolution_mat_zeropad(const double *m_in, const double *m_conv, double *m_res,
			size_t dim_w, size_t dim_h, size_t dim_conv_w, size_t dim_conv_h)
	{
		/* TODO */
	}
	
	void convolution_mat(const double *m_in, const double *m_conv, double *m_res,
			size_t dim_w, size_t dim_h, size_t dim_conv_w, size_t dim_conv_h)
	{
		/* TODO */
	}
}