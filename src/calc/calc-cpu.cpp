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

	void copy_vec(const double *v, double *vres, size_t dim)
	{
		const double *pres = v;
		while (v - pres < dim)
		{
			*vres = *v;
			v++;
			vres++;
		}
	}

	void apply_vec(const double *v, double *vres, size_t dim, std::function<double(double)> func)
	{
		const double *pres = v;
		while (v - pres < dim)
		{
			*vres = func(*v);
			v++;
			vres++;
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

		/* initialize to zero before addition */
		while (vres - p_vres < dim_v)
		{
			*vres = 0;
			vres++;
		}
		vres = p_vres;
		sum_vec_preserve(vset, vres, dim_v, num_v);
	}
	
	void sum_vec_preserve(const double *vset, double *vres, size_t dim_v, size_t num_v)
	{
		double *p_vres = vres;
		const double *p_vset = vset;

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

		if (m == mres)
		{
			/* TODO */
		}
		else
		{
			while (m - pm < dim_w * dim_h)
			{
				*mres = *m;
				m++;
				pmres--;
			}
		}

	}

	void convolution_mat(const double *m_in, const double *m_conv, double *m_res,
			int dim_w, int dim_h, int dim_conv_w, int dim_conv_h,
			const MatrixRange& range)
	{
        const int i_dim_w = dim_w;
        const int i_dim_h = dim_h;
	    for (int j=0;j<range.h;j++) {
            for (int i=0;i<range.w;i++) {
                double sum=0;
                for (int jj=0;jj<dim_conv_h;jj++) {
                    for (int ii=0;ii<dim_conv_w;ii++) {
                        int in_i = range.x + i + ii;
                        int in_j = range.y + j + jj;
                        if (in_i >= 0 && in_i < i_dim_w && in_j >= 0 && in_j < i_dim_h)
                            sum += m_conv[jj*dim_conv_w + ii] * m_in[in_j*dim_w + in_i];
                    }
                }
				m_res[j*range.w + i] = sum;
            }
	    }
	}
    
    void convolution_mat_no_zeros(const double *m_in, const double *m_conv, double *m_res,
			int dim_w, int dim_h, int dim_conv_w, int dim_conv_h)
	{
		convolution_mat(m_in, m_conv, m_res, dim_w, dim_h, dim_conv_w, dim_conv_h, MatrixRange(
			0, 0, dim_w - dim_conv_w + 1, dim_h - dim_conv_h + 1
		));
	}
    
    void convolution_mat_same_zeros(const double *m_in, const double *m_conv, double *m_res,
			int dim_w, int dim_h, int dim_conv_w, int dim_conv_h)
	{
		convolution_mat(m_in, m_conv, m_res, dim_w, dim_h, dim_conv_w, dim_conv_h, MatrixRange(
			-(dim_conv_w/2), -(dim_conv_h/2), dim_w, dim_h
		));
	}
    
    void convolution_mat_wide_zeros(const double *m_in, const double *m_conv, double *m_res,
			int dim_w, int dim_h, int dim_conv_w, int dim_conv_h)
	{
		convolution_mat(m_in, m_conv, m_res, dim_w, dim_h, dim_conv_w, dim_conv_h, MatrixRange(
			-dim_conv_w+1, -dim_conv_h+1, dim_w+dim_conv_w-1, dim_h+dim_conv_h-1
		));
	}
}
