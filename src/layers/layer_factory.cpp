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

    LayerFactory::LayerType LayerFactory::whatType(const LayerFactory::LayerSetting* set)
    {
        if (dynamic_cast<const SigmoidLayerSetting*>(set))
        {
            return LayerType::SIGMOID;
        }
        if (dynamic_cast<const ImageLayerSetting*>(set))
        {
            return LayerType::IMAGE;
        }
        if (dynamic_cast<const ConvLayerSetting*>(set))
        {
            return LayerType::CONVOLUTION;
        }
        if (dynamic_cast<const MaxPoolLayerSetting*>(set))
        {
            return LayerType::MAXPOOL;
        }
        return LayerType::NONE;
    }

    std::unique_ptr<Layer> LayerFactory::makeLayer(
            const LayerFactory::LayerSetting* prev_setting,
            const LayerFactory::LayerSetting* cur_setting)
    {
        auto prev_t = whatType(prev_setting);
        auto cur_t = whatType(cur_setting);

        auto creator = m_creators.find(std::make_pair(prev_t, cur_t));
        if (creator != m_creators.end())
        {
            return creator->second->create(*(prev_setting), *(cur_setting));
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
                cast_cur_set.learn_rate,
                cast_cur_set.dropout_rate,
                cast_cur_set.enable_dropout,
                cast_cur_set.uses_gpu,
                cast_cur_set.weight_decay
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
                cast_cur_set.learn_rate,
                cast_cur_set.dropout_rate,
                cast_cur_set.enable_dropout,
                cast_cur_set.uses_gpu,
                cast_cur_set.weight_decay
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
                cast_cur_set.learn_rate,
                cast_cur_set.dropout_rate,
                cast_cur_set.enable_dropout,
                cast_cur_set.uses_gpu,
                cast_cur_set.weight_decay
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
                    cast_prev_set.channel_num,
                    cast_cur_set.map_num,
                    cast_prev_set.image_w,
                    cast_prev_set.image_h,
                    cast_cur_set.recep_size,
                    cast_cur_set.learn_rate,
                    cast_cur_set.enable_zero_pad,
                    cast_cur_set.uses_gpu,
                    cast_cur_set.weight_decay
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
                    cast_prev_set.map_num,
                    cast_cur_set.map_num,
                    cast_prev_set.output_w,
                    cast_prev_set.output_h,
                    cast_cur_set.recep_size,
                    cast_cur_set.learn_rate,
                    cast_cur_set.enable_zero_pad,
                    cast_cur_set.uses_gpu,
                    cast_cur_set.weight_decay
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
                    cast_prev_set.map_num,
                    cast_cur_set.map_num,
                    cast_prev_set.output_w,
                    cast_prev_set.output_h,
                    cast_cur_set.recep_size,
                    cast_cur_set.learn_rate,
                    cast_cur_set.enable_zero_pad,
                    cast_cur_set.uses_gpu,
                    cast_cur_set.weight_decay
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
                    cast_cur_set.map_num,
                    cast_cur_set.input_w,
                    cast_cur_set.input_h,
                    cast_cur_set.pool_w,
                    cast_cur_set.pool_h,
                    cast_cur_set.stride,
                    cast_cur_set.uses_gpu
                })
        );
    }

    void LayerFactory::getOutputDimension(const LayerSetting* set,
            size_t& width, size_t& height)
    {
        auto set_t = whatType(set);
        if (set_t == LayerType::IMAGE)
        {
            auto ils = static_cast<const ImageLayerSetting&>(*(set));
            width = ils.image_w;
            height = ils.image_h;
        }
        else if (set_t == LayerType::CONVOLUTION)
        {
            auto cs = static_cast<const ConvLayerSetting&>(*(set));
            width = cs.output_w;
            height = cs.output_h;
        }
        else if (set_t == LayerType::MAXPOOL)
        {
            auto mps = static_cast<const MaxPoolLayerSetting&>(*(set));
            width = mps.output_w;
            height = mps.output_h;
        }
    }

    size_t LayerFactory::getMapNum(const LayerSetting* set)
    {
        size_t num = 0;
        auto set_t = whatType(set);
        if (set_t == LayerType::IMAGE)
        {
            auto ils = static_cast<const ImageLayerSetting&>(*(set));
            num = ils.channel_num;
        }
        else if (set_t == LayerType::CONVOLUTION)
        {
            auto cs = static_cast<const ConvLayerSetting&>(*(set));
            num = cs.map_num;
        }
        else if (set_t == LayerType::MAXPOOL)
        {
            auto mps = static_cast<const MaxPoolLayerSetting&>(*(set));
            num = mps.map_num;
        }
        return num;
    }
}
