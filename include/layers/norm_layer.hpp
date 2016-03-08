#ifndef __NORM_LAYER_HPP
#define __NORM_LAYER_HPP

#include "layers/layer.hpp"
#include "layers/conv_layer.hpp"

namespace NeuralNet
{
    class NormalizeLayer: public Layer
    {
    public:
        using ActivationFunc = ConvLayer::ActivationFunc;

        struct Setting
        {
            size_t map_num;
            size_t input_width;
            size_t input_height;
            double halfwidth;
            double alpha;
            double beta;
            ActivationFunc func;
        };

        NormalizeLayer(const Setting& set);
        virtual ~NormalizeLayer();

        virtual void forward_cpu(const LayerData& prev, LayerData& current);
        virtual void forward_gpu(const LayerData& prev, LayerData& current);
        virtual void backward_cpu(LayerData& prev, LayerData& current);
        virtual void backward_gpu(LayerData& prev, LayerData& current);

        virtual std::unique_ptr<LayerData> createLayerData(size_t train_num);

        virtual void importLayer(const Json::Value& coeffs);
        virtual Json::Value exportLayer();

        virtual std::string what() { return "lr_normalize"; }
        virtual size_t getNeuronNum() const;

    private:
        const Setting m_set;
        const size_t m_output_neurons;
    };
}

#endif // __NORM_LAYER_HPP
