#ifndef __LAYER_HPP
#define __LAYER_HPP

#include <cstdlib>
#include <memory>
#include <string>
#include "layers/layer_data.hpp"
#include "json/json.h"

namespace NeuralNet
{
	/**
	 * interface representing a layer.
	 */
	class Layer
	{
    public:
		virtual ~Layer() {}

		/* forwarding input of the layer */
		void forward(const LayerData& prev, LayerData& current)
		{
#ifdef USES_GPU
			forward_gpu(prev, current);
#else
			forward_cpu(prev, current);
#endif
		}

		/* backpropagation of the layer */
		void backward(LayerData& prev, LayerData& current)
		{
#ifdef USES_GPU
			backward_gpu(prev, current);
#else
			backward_cpu(prev, current);
#endif
		}
		
		/* creation of appropriate layer data for the layer */
		virtual std::unique_ptr<LayerData> createLayerData() = 0;
		
		/* import/export of layer coefficients.
		 * importLayer() may emit Json::Exception during execution
		 */
		virtual void importLayer(const Json::Value& coeffs) = 0;
		virtual Json::Value exportLayer() = 0;
		
		/* description of the layer */
		virtual std::string what() = 0;
		
	protected:
		/**
		 * CPU and GPU versions of the forward() and backward() that child classes
		 * need to implement
		 */
		virtual void forward_cpu(const LayerData& prev, LayerData& current) = 0;
		virtual void forward_gpu(const LayerData& prev, LayerData& current) = 0;
		virtual void backward_cpu(LayerData& prev, LayerData& current) = 0;
		virtual void backward_gpu(LayerData& prev, LayerData& current) = 0;
	};
}

#endif // __LAYER_HPP
