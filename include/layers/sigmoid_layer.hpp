#ifndef __SIGMOID_LAYER_HPP
#define __SIGMOID_LAYER_HPP

#include "layers/layer.hpp"
#include <cmath>
#include <functional>

namespace NeuralNet
{
	class SigmoidLayer: public Layer
	{
    public:
        virtual ~SigmoidLayer() {}

		virtual void forward_cpu(const LayerData& prev, LayerData& current);
		virtual void forward_gpu(const LayerData& prev, LayerData& current);
		virtual void backward_cpu(LayerData& prev, LayerData& current);
		virtual void backward_gpu(LayerData& prev, LayerData& current);
		
		static const std::function<double(double)> f_sigmoid;
		static const std::function<double(double)> f_sigmoid_prime;
	};
}

#endif // __SIGMOID_LAYER_HPP
