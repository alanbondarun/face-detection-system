#ifndef __CONV_LAYER_HPP
#define __CONV_LAYER_HPP

#include "layers/layer.hpp"
#include "layers/layer_data.hpp"
#include <cstdlib>

namespace NeuralNet
{
	class ConvLayer: public Layer
	{
	public:
		struct Dimension
		{
			size_t prev_map_num;
			size_t current_map_num;
			size_t image_width;
			size_t image_height;
			size_t recep_size;
		};
		
		enum class ActivationFunc
		{
			SIGMOID, RELU,
		};
	
		ConvLayer(const Dimension& dim, ActivationFunc func);
        virtual ~ConvLayer();

		virtual void forward_cpu(const LayerData& prev, LayerData& current);
		virtual void forward_gpu(const LayerData& prev, LayerData& current);
		virtual void backward_cpu(LayerData& prev, LayerData& current);
		virtual void backward_gpu(LayerData& prev, LayerData& current);
		
		virtual void std::unique_ptr<LayerData> createLayerData();
		
	private:
		static const double eta;
		
		Dimension m_dim;
		const std::function<double(double)> f_activation;
		const std::function<double(double)> f_activation_prime;
		double *m_weight;
		double *m_bias;
	};
}

#endif // __CONV_LAYER_HPP