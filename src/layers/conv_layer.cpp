#include "layers/conv_layer.hpp"
#include "calc/calc-cpu.hpp"
#include "calc/util-functions.hpp"
#include "utils/make_unique.hpp"

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
			f_convolution = convolution_mat_zeropad;
			m_output_width = m_set.image_width;
			m_output_height = m_set.image_height;
		}
		else
		{
			f_convolution = convolution_mat;
			m_output_width = m_set.image_width - (m_set.recep_size - 1);
			m_output_height = m_set.image_height - (m_set.recep_size - 1);
		}
		
		m_weight = new double[m_set.current_map_num * m_set.prev_map_num
				* m_set.recep_size * m_set.recep_size];
	}
	
	ConvLayer::~ConvLayer()
	{
		delete [] m_weight;
	}
	
	void ConvLayer::forward_cpu(const LayerData& prev, LayerData& current)
	{
		auto prev_a = prev.get(LayerData::DataIndex::ACTIVATION);
		auto prev_z = prev.get(LayerData::DataIndex::INTER_VALUE);
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
}