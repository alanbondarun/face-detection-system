#include "layers/layer_factory.hpp"
#include "layers/layer.hpp"
#include "layers/sigmoid_layer.hpp"
#include "layers/conv_layer.hpp"
#include "layers/max_pool_layer.hpp"
#include "utils/make_unique.hpp"
#include <memory>
#include <utility>

namespace NeuralNet
{
	LayerFactory::LayerFactory()
	{
		/* initialization of creator classes */
		m_creators[std::make_pair(LayerType::SIGMOID, LayerType::SIGMOID)]
				= std::move(std::make_unique< LayerCreator<SigmoidLayerSetting, SigmoidLayerSetting> >());
		m_creators[std::make_pair(LayerType::CONVOLUTION, LayerType::SIGMOID)]
				= std::move(std::make_unique< LayerCreator<ConvLayerSetting, SigmoidLayerSetting> >());
		m_creators[std::make_pair(LayerType::MAXPOOL, LayerType::SIGMOID)]
				= std::move(std::make_unique< LayerCreator<MaxPoolLayerSetting, SigmoidLayerSetting> >());
		m_creators[std::make_pair(LayerType::IMAGE, LayerType::CONVOLUTION)]
				= std::move(std::make_unique< LayerCreator<ImageLayerSetting, ConvLayerSetting> >());
		m_creators[std::make_pair(LayerType::CONVOLUTION, LayerType::CONVOLUTION)]
				= std::move(std::make_unique< LayerCreator<ConvLayerSetting, ConvLayerSetting> >());
		m_creators[std::make_pair(LayerType::MAXPOOL, LayerType::CONVOLUTION)]
				= std::move(std::make_unique< LayerCreator<MaxPoolLayerSetting, ConvLayerSetting> >());
		m_creators[std::make_pair(LayerType::CONVOLUTION, LayerType::MAXPOOL)]
				= std::move(std::make_unique< LayerCreator<ConvLayerSetting, MaxPoolLayerSetting> >());
	}

	std::unique_ptr<Layer> LayerFactory::makeLayer(const SettingPair& prev_setting,
			const SettingPair& cur_setting)
	{
		auto creator = m_creators.find(std::make_pair(prev_setting.first, cur_setting.first));
		if (creator != m_creators.end())
		{
			return creator->second->create(*(prev_setting.second), *(cur_setting.second));
		}
		return std::unique_ptr<Layer>();
	}

	template <>
	std::unique_ptr<Layer> LayerCreator<LayerFactory::SigmoidLayerSetting,
			LayerFactory::SigmoidLayerSetting>::create(
			const LayerFactory::LayerSetting& prev_set,
			const LayerFactory::LayerSetting& cur_set)
	{
		auto& cast_prev_set = static_cast<const LayerFactory::SigmoidLayerSetting&>(prev_set);
		auto& cast_cur_set = static_cast<const LayerFactory::SigmoidLayerSetting&>(cur_set);
		return std::make_unique<SigmoidLayer>(
			SigmoidLayer::Setting({
				cast_prev_set.neuron_num,
				cast_cur_set.neuron_num,
				cast_cur_set.train_num,
				cast_cur_set.learn_rate,
				cast_cur_set.dropout_rate,
				cast_cur_set.enable_dropout
			})
		);
	}

	template <>
	std::unique_ptr<Layer> LayerCreator<LayerFactory::ConvLayerSetting,
			LayerFactory::SigmoidLayerSetting>::create(
			const LayerFactory::LayerSetting& prev_set,
			const LayerFactory::LayerSetting& cur_set)
	{
		auto& cast_prev_set = static_cast<const LayerFactory::ConvLayerSetting&>(prev_set);
		auto& cast_cur_set = static_cast<const LayerFactory::SigmoidLayerSetting&>(cur_set);
		return std::make_unique<SigmoidLayer>(
			SigmoidLayer::Setting({
				cast_prev_set.map_num * cast_prev_set.output_w * cast_prev_set.output_h,
				cast_cur_set.neuron_num,
				cast_cur_set.train_num,
				cast_cur_set.learn_rate,
				cast_cur_set.dropout_rate,
				cast_cur_set.enable_dropout
			})
		);
	}

	template <>
	std::unique_ptr<Layer> LayerCreator<LayerFactory::MaxPoolLayerSetting,
			LayerFactory::SigmoidLayerSetting>::create(
			const LayerFactory::LayerSetting& prev_set,
			const LayerFactory::LayerSetting& cur_set)
	{
		auto& cast_prev_set = static_cast<const LayerFactory::MaxPoolLayerSetting&>(prev_set);
		auto& cast_cur_set = static_cast<const LayerFactory::SigmoidLayerSetting&>(cur_set);
		return std::make_unique<SigmoidLayer>(
			SigmoidLayer::Setting({
				cast_prev_set.map_num * cast_prev_set.output_w * cast_prev_set.output_h,
				cast_cur_set.neuron_num,
				cast_cur_set.train_num,
				cast_cur_set.learn_rate,
				cast_cur_set.dropout_rate,
				cast_cur_set.enable_dropout
			})
		);
	}

	template <>
	std::unique_ptr<Layer> LayerCreator<LayerFactory::ImageLayerSetting,
			LayerFactory::ConvLayerSetting>::create(
			const LayerFactory::LayerSetting& prev_set,
			const LayerFactory::LayerSetting& cur_set)
	{
		auto& cast_prev_set = static_cast<const LayerFactory::ImageLayerSetting&>(prev_set);
		auto& cast_cur_set = static_cast<const LayerFactory::ConvLayerSetting&>(cur_set);
		return std::make_unique<ConvLayer>(
				ConvLayer::LayerSetting({
					cast_cur_set.train_num,
					cast_prev_set.channel_num,
					cast_cur_set.map_num,
					cast_prev_set.image_w,
					cast_prev_set.image_h,
					cast_cur_set.recep_size,
					cast_cur_set.learn_rate,
					cast_cur_set.enable_zero_pad
				}),
				ConvLayer::ActivationFunc::RELU
		);
	}

	template <>
	std::unique_ptr<Layer> LayerCreator<LayerFactory::ConvLayerSetting,
			LayerFactory::ConvLayerSetting>::create(
			const LayerFactory::LayerSetting& prev_set,
			const LayerFactory::LayerSetting& cur_set)
	{
		auto& cast_prev_set = static_cast<const LayerFactory::ConvLayerSetting&>(prev_set);
		auto& cast_cur_set = static_cast<const LayerFactory::ConvLayerSetting&>(cur_set);
		return std::make_unique<ConvLayer>(
				ConvLayer::LayerSetting({
					cast_cur_set.train_num,
					cast_prev_set.map_num,
					cast_cur_set.map_num,
					cast_prev_set.output_w,
					cast_prev_set.output_h,
					cast_cur_set.recep_size,
					cast_cur_set.learn_rate,
					cast_cur_set.enable_zero_pad
				}),
				ConvLayer::ActivationFunc::RELU
		);
	}

	template <>
	std::unique_ptr<Layer> LayerCreator<LayerFactory::MaxPoolLayerSetting,
			LayerFactory::ConvLayerSetting>::create(
			const LayerFactory::LayerSetting& prev_set,
			const LayerFactory::LayerSetting& cur_set)
	{
		auto& cast_prev_set = static_cast<const LayerFactory::MaxPoolLayerSetting&>(prev_set);
		auto& cast_cur_set = static_cast<const LayerFactory::ConvLayerSetting&>(cur_set);
		return std::make_unique<ConvLayer>(
				ConvLayer::LayerSetting({
					cast_cur_set.train_num,
					cast_prev_set.map_num,
					cast_cur_set.map_num,
					cast_prev_set.output_w,
					cast_prev_set.output_h,
					cast_cur_set.recep_size,
					cast_cur_set.learn_rate,
					cast_cur_set.enable_zero_pad
				}),
				ConvLayer::ActivationFunc::RELU
		);
	}

	template <>
	std::unique_ptr<Layer> LayerCreator<LayerFactory::ConvLayerSetting,
			LayerFactory::MaxPoolLayerSetting>::create(
			const LayerFactory::LayerSetting& prev_set,
			const LayerFactory::LayerSetting& cur_set)
	{
		auto& cast_cur_set = static_cast<const LayerFactory::MaxPoolLayerSetting&>(cur_set);
		return std::make_unique<MaxPoolLayer>(
				MaxPoolLayer::Dimension({
					cast_cur_set.train_num,
					cast_cur_set.map_num,
					cast_cur_set.input_w,
					cast_cur_set.input_h,
					cast_cur_set.pool_w,
					cast_cur_set.pool_h,
				})
		);
	}

	void LayerFactory::getOutputDimension(const SettingPair& set_pair,
			size_t& width, size_t& height)
	{
		if (set_pair.first == LayerType::IMAGE)
		{
			auto ils = static_cast<const ImageLayerSetting&>(*(set_pair.second));
			width = ils.image_w;
			height = ils.image_h;
		}
		else if (set_pair.first == LayerType::CONVOLUTION)
		{
			auto cs = static_cast<const ConvLayerSetting&>(*(set_pair.second));
			width = cs.output_w;
			height = cs.output_h;
		}
		else if (set_pair.first == LayerType::MAXPOOL)
		{
			auto mps = static_cast<const MaxPoolLayerSetting&>(*(set_pair.second));
			width = mps.output_w;
			height = mps.output_h;
		}
	}
	
	size_t LayerFactory::getMapNum(const SettingPair& set_pair)
	{
		size_t num = 0;
		if (set_pair.first == LayerType::IMAGE)
		{
			auto ils = static_cast<const ImageLayerSetting&>(*(set_pair.second));
			num = ils.channel_num;
		}
		else if (set_pair.first == LayerType::CONVOLUTION)
		{
			auto cs = static_cast<const ConvLayerSetting&>(*(set_pair.second));
			num = cs.map_num;
		}
		else if (set_pair.first == LayerType::MAXPOOL)
		{
			auto mps = static_cast<const MaxPoolLayerSetting&>(*(set_pair.second));
			num = mps.map_num;
		}
		return num;
	}
}