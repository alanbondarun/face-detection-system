#include "layers/conv_layer.hpp"
#include "calc/calc-cpu.hpp"
#include "calc/util-functions.hpp"
#include "utils/make_unique.hpp"
#include <random>
#include <cstring>
#include <cmath>

namespace NeuralNet
{
	const double ConvLayer::eta = 0.1;
	
	ConvLayer::ConvLayer(const LayerSetting& setting, ActivationFunc func)
		: m_set(setting)
	{
		switch (func)
		{
		case ActivationFunc::SIGMOID:
			f_activation = f_sigmoid;
			f_activation_prime = f_sigmoid_prime;
			break;
		case ActivationFunc::RELU:
			f_activation = f_relu;
			f_activation_prime = f_relu_prime;
			break;
		}
		
		if (m_set.enable_zero_pad)
		{
			f_convolution = convolution_mat_same_zeros;
			f_convol_back = convolution_mat_same_zeros;
			m_output_width = m_set.image_width;
			m_output_height = m_set.image_height;
		}
		else
		{
			f_convolution = convolution_mat_no_zeros;
			f_convol_back = convolution_mat_wide_zeros;
			m_output_width = m_set.image_width - (m_set.recep_size - 1);
			m_output_height = m_set.image_height - (m_set.recep_size - 1);
		}

		const size_t num_weights = m_set.current_map_num * m_set.prev_map_num
				* m_set.recep_size * m_set.recep_size;
		m_weight = new double[num_weights];

		/* weight initialization */
		std::random_device rd;
		std::mt19937 rgen(rd());
		std::normal_distribution<double> dist_w(0.0, std::sqrt(1.0 / m_output_width * m_output_height));

		for (size_t i = 0; i < num_weights; i++)
			m_weight[i] = dist_w(rgen);
	}
	
	ConvLayer::~ConvLayer()
	{
		delete [] m_weight;
	}
	
	void ConvLayer::forward_cpu(const LayerData& prev, LayerData& current)
	{
		auto prev_a = prev.get(LayerData::DataIndex::ACTIVATION);
		auto cur_a = current.get(LayerData::DataIndex::ACTIVATION);
		auto cur_z = current.get(LayerData::DataIndex::INTER_VALUE);
		
		memset(cur_z, 0, sizeof(double) * m_set.train_num
				* m_set.current_map_num * m_output_width * m_output_height);
		double *temp_z = new double[m_output_width * m_output_height];
		
		for (size_t i=0; i<m_set.train_num; i++)
		{		
			size_t w_offset = 0;
			size_t prev_offset = 0;
			size_t cur_offset = i * m_set.current_map_num * m_output_width * m_output_height * sizeof(double);
			
			for (size_t ncur = 0; ncur < m_set.current_map_num; ncur++)
			{
				prev_offset = i * m_set.prev_map_num * m_set.image_width * m_set.image_height * sizeof(double);
				for (size_t nprev = 0; nprev < m_set.prev_map_num; nprev++)
				{
					f_convolution(
						prev_a + prev_offset, m_weight + w_offset, temp_z,
						m_set.image_width, m_set.image_height, m_set.recep_size, m_set.recep_size
					);
					add_vec(cur_z + cur_offset, temp_z, cur_z + cur_offset, m_output_width * m_output_height);
					
					w_offset += (m_set.recep_size * m_set.recep_size * sizeof(double));
					prev_offset += (m_set.image_width * m_set.image_height * sizeof(double));
				}
				
				apply_vec(cur_z + cur_offset, cur_a + cur_offset, m_output_width * m_output_height,
						f_activation);
				cur_offset += (m_output_width * m_output_height * sizeof(double));
			}
			
		}
		
		delete [] temp_z;
	}
	
	void ConvLayer::forward_gpu(const LayerData& prev, LayerData& current)
	{
		/* TODO: OpenCL intergration */
	}
	
	void ConvLayer::backward_cpu(LayerData& prev, LayerData& current)
	{
		const auto train_num = m_set.train_num;
		const int i_recep_size = m_set.recep_size;
		auto prev_a = prev.get(LayerData::DataIndex::ACTIVATION);
		auto prev_z = prev.get(LayerData::DataIndex::INTER_VALUE);
		auto prev_e = prev.get(LayerData::DataIndex::ERROR);
		auto cur_e = current.get(LayerData::DataIndex::ERROR);
		
		memset(prev_e, 0, sizeof(double) * m_set.train_num
				* m_set.prev_map_num * m_set.image_width * m_set.image_height);	
				
		double *sprime_z = new double[m_set.train_num * m_set.prev_map_num
				* m_set.image_width * m_set.image_height];
		apply_vec(prev_z, sprime_z, m_set.train_num * m_set.prev_map_num
				* m_set.image_width * m_set.image_height,
				f_activation_prime);
		
		/* calculate error value for previous layer */
		double *temp_w = new double[m_set.recep_size * m_set.recep_size];
		double *temp_pe = new double[m_set.image_width * m_set.image_height];
		for (size_t i=0; i<train_num; i++)
		{
			size_t w_offset = 0;
			size_t cur_offset = 0;
			size_t prev_offset = i * m_set.prev_map_num * m_set.image_width * m_set.image_height * sizeof(double);;
			for (size_t nprev = 0; nprev < m_set.prev_map_num; nprev++)
			{
				cur_offset = i * m_set.current_map_num * m_output_width * m_output_height * sizeof(double);
				w_offset = nprev * m_set.recep_size * m_set.recep_size;
				
				for (size_t ncur = 0; ncur < m_set.current_map_num; ncur++)
				{
					flip_mat(m_weight + w_offset, temp_w, m_set.recep_size, m_set.recep_size);
					f_convol_back(
						cur_e + cur_offset, temp_w, temp_pe,
						m_output_width, m_output_height, m_set.recep_size, m_set.recep_size
					);
					add_vec(prev_e + prev_offset, temp_pe, prev_e + prev_offset,
							m_set.image_width * m_set.image_height);
					
					w_offset += (m_set.prev_map_num * m_set.recep_size * m_set.recep_size * sizeof(double));
					cur_offset += (m_output_width * m_output_height * sizeof(double));
				}
				
				pmul_vec(prev_e + prev_offset, sprime_z + prev_offset, prev_e + prev_offset,
						m_set.image_width * m_set.image_height);
				
				prev_offset += (m_set.image_width * m_set.image_height * sizeof(double));
			}
		}
		
		/* calculate delta_w and update current weight */
		double *delta_w = new double[m_set.recep_size * m_set.recep_size];
		memset(delta_w, 0, sizeof(double) * m_set.recep_size * m_set.recep_size);
		size_t dw_offset = 0;
		for (size_t ncur = 0; ncur < m_set.current_map_num; ncur++)
		{
			for (size_t nprev = 0; nprev < m_set.prev_map_num; nprev++)
			{
				size_t cur_offset = ncur * m_output_width * m_output_height * sizeof(double);
				size_t prev_offset = (nprev * m_set.image_width * m_set.image_height * sizeof(double));
				
				memset(delta_w, 0, sizeof(double) * m_set.recep_size * m_set.recep_size);
				
				for (size_t i = 0; i < m_set.train_num; i++)
				{
					if (m_set.enable_zero_pad)
					{
						convolution_mat_no_zeros(prev_a + prev_offset, cur_e + cur_offset, temp_w,
							m_set.image_width, m_set.image_height, m_output_width, m_output_height);
					}
					else
					{
						convolution_mat(prev_a + prev_offset, cur_e + cur_offset, temp_w,
							m_set.image_width, m_set.image_height, m_output_width, m_output_height,
							MatrixRange(-(i_recep_size/2), -(i_recep_size/2), i_recep_size, i_recep_size));
					}
					add_vec(delta_w, temp_w, delta_w, m_set.recep_size * m_set.recep_size);
					
					prev_offset += (m_set.prev_map_num * m_set.image_width * m_set.image_height * sizeof(double));
					cur_offset += (m_set.current_map_num * m_output_width * m_output_height * sizeof(double));
				}
				
				apply_vec(delta_w, delta_w, m_set.recep_size * m_set.recep_size, [train_num](double in) -> double {
					return -in*eta/train_num;
				});
				add_vec(m_weight + dw_offset, delta_w, m_weight + dw_offset,
						m_set.recep_size * m_set.recep_size);
				
				dw_offset += (m_set.recep_size * m_set.recep_size);
			}
		}
		
		delete [] delta_w;
		delete [] temp_pe;
		delete [] temp_w;
		delete [] sprime_z;
	}
	
	void ConvLayer::backward_gpu(LayerData& prev, LayerData& current)
	{
		/* TODO: OpenCL intergration */
	}
	
	std::unique_ptr<LayerData> ConvLayer::createLayerData()
	{
		return std::make_unique<LayerData>(
			m_set.train_num,
			m_set.current_map_num * m_output_width * m_output_height
		);
	}
	
	void ConvLayer::importLayer(const Json::Value& coeffs)
	{
		auto coeff_dim = coeffs["dimension"];
		size_t cur_maps = coeff_dim["map_num"].asUInt();
		if (cur_maps != m_set.current_map_num)
			throw Json::LogicError("invalid map_num");
		size_t recep_size = coeff_dim["recep_size"].asUInt();
		if (recep_size != m_set.recep_size)
			throw Json::LogicError("invalid recep_size");
		
		size_t weight_offset = 0;
		auto weight_lists = coeffs["weight"];
		if (weight_lists.size() != m_set.current_map_num * m_set.prev_map_num)
			throw Json::LogicError("invalid number of feature maps");
		for (int i = 0; i < m_set.current_map_num * m_set.prev_map_num; i++)
		{
			auto weights = weight_lists[i];
			if (weights.size() != m_set.recep_size * m_set.recep_size)
				throw Json::LogicError("invalid feature map");
			
			for (int j = 0; j < m_set.recep_size * m_set.recep_size; j++)
			{
				*(m_weight + weight_offset) = weights[j].asDouble();
				weight_offset++;
			}
		}
	}
	
	Json::Value ConvLayer::exportLayer()
	{
		Json::Value coeff_value(Json::objectValue);

		Json::Value coeff_dim(Json::objectValue);
		coeff_dim["map_num"] = Json::Value(static_cast<Json::UInt>(m_set.current_map_num));
		coeff_dim["recep_size"] = Json::Value(static_cast<Json::UInt>(m_set.recep_size));
		coeff_value["dimension"] = coeff_dim;

		size_t weight_offset = 0;
		Json::Value weight_lists(Json::arrayValue);
		for (size_t i = 0; i < m_set.current_map_num * m_set.prev_map_num; i++)
		{
			Json::Value weights(Json::arrayValue);
			for (size_t j = 0; j < m_set.recep_size * m_set.recep_size; j++)
			{
				weights.append(Json::Value(m_weight[weight_offset]));
				weight_offset++;
			}

			weight_lists.append(weights);
		}
		coeff_value["weight"] = weight_lists;

		return coeff_value;
	}
}