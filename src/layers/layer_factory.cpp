#include "layers/layer_factory.hpp"
#include <memory>

namespace NeuralNet
{
	std::unique_ptr<Layer> LayerFactory::makeLayer(LayerType type,
			const LayerSetting& prev_setting, const LayerSetting& cur_setting)
	{
		switch (type)
		{
		case LayerType::SIGMOID:
			break;
		case LayerType::CONVOLUTION:
			break;
		case LayerType::MAXPOOL:
			break;
		default:
			return std::unique_ptr<Layer>();
		}
	}
}