#ifndef __LAYER_FACTORY_HPP
#define __LAYER_FACTORY_HPP

#include <utility>
#include <vector>
#include <memory>
#include "layers/layer.hpp"

namespace NeuralNet
{
	class LayerFactory
	{
	public:
		enum class LayerType
		{
			SIGMOID, CONVOLUTION, MAXPOOL, BRANCH
		};
		
		/* layer setting structs */
		struct LayerSetting
		{
			size_t train_num;
			explicit LayerSetting() : train_num(0) {}
			explicit LayerSetting(size_t _t) : train_num(_t) {}
		};
		struct SigmoidLayerSetting: public LayerSetting
		{
			size_t neuron_num;
			double learn_rate;
			explicit SigmoidLayerSetting(size_t _t, size_t _n, double _l)
				: LayerSetting(_t), neuron_num(_n), learn_rate(_l) {}
		};
		struct ImageLayerSetting: public LayerSetting
		{
			size_t image_w, image_h, map_num;
			explicit ImageLayerSetting(size_t _t, size_t _w, size_t _h, size_t _m)
				: LayerSetting(_t), image_w(_w), image_h(_h), map_num(_m) {}
		};
		struct ConvLayerSetting: public LayerSetting
		{
			size_t map_num, recep_size;
			double learn_rate;
			bool enable_zero_pad;
			explicit ConvLayerSetting(size_t _t, size_t _m, size_t _r, double _l, bool _zeropad)
				: LayerSetting(_t), map_num(_m), recep_size(_r), learn_rate(_l),
				enable_zero_pad(_zeropad) {}
		};
		struct MaxPoolLayerSetting: public LayerSetting
		{
			size_t pool_w, pool_h;
			explicit MaxPoolLayerSetting(size_t _t, size_t _w, size_t _h)
				: LayerSetting(_t), pool_w(_w), pool_h(_h) {}
		};
		
		/* returns empty unique_ptr for invalid layer type */
		static std::unique_ptr<Layer> makeLayer(LayerType type,
				const LayerSetting& prev_setting,
				const LayerSetting& cur_setting);
	};
}

#endif // __LAYER_FACTORY_HPP