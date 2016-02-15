#include "layers/conv_layer.hpp"
#include "calc/calc-cpu.hpp"
#include "calc/util-functions.hpp"
#include "utils/make_unique.hpp"

namespace NeuralNet
{
	const double ConvLayer::eta = 0.1;
	
	ConvLayer::ConvLayer(const Dimension& dim, ActivationFunc func)
		: m_dim(dim)
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
		
		m_weight = new double[dim.prev_map_num * dim.current_map_num
				* dim.recep_size * dim.recep_size];
		m_bias = new double[dim.current_map_num * dim.recep_size
				* dim.recep_size];
	}
	
	ConvLayer::~ConvLayer()
	{
		delete [] m_bias;
		delete [] m_weight;
	}
	
	void ConvLayer::forward_cpu(const LayerData& prev, LayerData& current)
	{
		
	}
	
	void forward_gpu(const LayerData& prev, LayerData& current)
	{
		/* TODO: OpenCL intergration */
	}
	
	void backward_cpu(LayerData& prev, LayerData& current)
	{
		
	}
	
	void backward_gpu(LayerData& prev, LayerData& current)
	{
		/* TODO: OpenCL intergration */
	}
	
	std::unique_ptr<LayerData> ConvLayer::createLayerData()
	{
		return std::make_unique<LayerData>(
			m_dim.train_num,
			
		);
	}
}