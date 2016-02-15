#ifndef __MAX_POOL_LAYER_HPP
#define __MAX_POOL_LAYER_HPP

#include "layers/layer_data.hpp"
#include "layers/layer.hpp"
#include <cstdlib>

namespace NeuralNet
{
	class MaxPoolLayer: public Layer
	{
	public:
		struct Dimension
		{
			size_t train_num;
			size_t map_num;
			size_t image_width;
			size_t image_height;
			size_t pool_width;
			size_t pool_height;
		};
	
		MaxPoolLayer(const Dimension& dim);
        virtual ~MaxPoolLayer() {}

		virtual void forward_cpu(const LayerData& prev, LayerData& current);
		virtual void forward_gpu(const LayerData& prev, LayerData& current);
		virtual void backward_cpu(LayerData& prev, LayerData& current);
		virtual void backward_gpu(LayerData& prev, LayerData& current);
		
		virtual std::unique_ptr<LayerData> createLayerData();
		
	private:
		const Dimension m_dim;
		const size_t m_output_width, m_output_height;
	};
}

#endif