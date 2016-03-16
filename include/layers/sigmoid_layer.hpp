#ifndef __SIGMOID_LAYER_HPP
#define __SIGMOID_LAYER_HPP

#include "layers/layer.hpp"
#include "json/json.h"
#include <functional>

namespace NeuralNet
{
    class SigmoidLayer: public Layer
    {
    public:
        struct Setting
        {
            size_t prev_neurons;
            size_t current_neurons;
            float learn_rate;
            float dropout_rate;
            bool dropout_enable;
            bool uses_gpu;
        };
        SigmoidLayer(const Setting& set);
        virtual ~SigmoidLayer();

        virtual void forward_cpu(const LayerData& prev, LayerData& current);
        virtual void forward_gpu(const LayerData& prev, LayerData& current);
        virtual void backward_cpu(LayerData& prev, LayerData& current);
        virtual void backward_gpu(LayerData& prev, LayerData& current);

        virtual std::unique_ptr<LayerData> createLayerData(size_t train_num);

        virtual void importLayer(const Json::Value& coeffs);
        virtual Json::Value exportLayer();

        virtual std::string what() { return "sigmoid"; }
        virtual size_t getNeuronNum() const;

        void setDropout(bool enable) { m_dropout_enabled = enable; }

        static const std::function<float(float)> f_sigmoid;
        static const std::function<float(float)> f_sigmoid_prime;

    private:
        const size_t m_prev_d, m_current_d;
        float m_learn_rate;

        const bool m_uses_dropout;
        bool m_dropout_enabled;
        const float m_dropout_rate;
        const bool m_uses_gpu;

        float *m_weight;
        float *m_bias;
        float *m_dropout_coeff;

    public:
        virtual void setLearnRate(float rate) { m_learn_rate = rate; }
        virtual float getLearnRate() const { return m_learn_rate; }
    };
}

#endif // __SIGMOID_LAYER_HPP
