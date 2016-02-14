#ifndef __SIGMOID_LAYER_HPP
#define __SIGMOID_LAYER_HPP

#include "layers/layer.hpp"
#include <functional>

namespace NeuralNet
{
	class SigmoidLayer: public Layer
	{
    public:
		SigmoidLayer(size_t prev_neurons, size_t current_neurons);
        virtual ~SigmoidLayer();

		virtual void forward_cpu(const LayerData& prev, LayerData& current);
		virtual void forward_gpu(const LayerData& prev, LayerData& current);
		virtual void backward_cpu(LayerData& prev, LayerData& current);
		virtual void backward_gpu(LayerData& prev, LayerData& current);
		
		static const std::function<double(double)> f_sigmoid;
		static const std::function<double(double)> f_sigmoid_prime;
		
	private:
		static const double eta;
		
		size_t m_prev_d, m_current_d;
		double *m_weight;
		double *m_bias;
	};
}

#endif // __SIGMOID_LAYER_HPP
