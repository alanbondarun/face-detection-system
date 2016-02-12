#ifndef __SIGMOID_LAYER_HPP
#define __SIGMOID_LAYER_HPP

#include "layers/layer.hpp"
#include <cmath>

namespace NeuralNet
{
	class SigmoidLayer: public Layer
	{
    public:
        virtual ~SigmoidLayer() {}

		virtual void forward_cpu(const LayerData& prev, LayerData& current);
		virtual void forward_gpu(const LayerData& prev, LayerData& current);
		virtual void backward_cpu(LayerData& prev, const LayerData& current);
		virtual void backward_gpu(LayerData& prev, const LayerData& current);

	private:
		/* calculating sigmoid function */
		inline double sigmoid(const double in)
		{
		    return 1.0 / (1.0 + std::exp(-in));
		}
	};
}

#endif // __SIGMOID_LAYER_HPP
