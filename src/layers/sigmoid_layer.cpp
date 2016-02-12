#include "layers/sigmoid_layer.hpp"
#include <random>

namespace NeuralNet
{
	void SigmoidLayer::forward_cpu(const LayerData& prev, LayerData& current)
	{
		
	}
	
	void SigmoidLayer::forward_gpu(const LayerData& prev, LayerData& current)
	{
		/* TODO */
	}
	
	void SigmoidLayer::backward_cpu(LayerData& prev, const LayerData& current)
	{
		
	}
	
	void SigmoidLayer::backward_gpu(LayerData& prev, const LayerData& current)
	{
		/* TODO */
	}
}
