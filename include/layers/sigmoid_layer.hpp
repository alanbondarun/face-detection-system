#ifndef __SIGMOID_LAYER_HPP
#define __SIGMOID_LAYER_HPP

#include "layers/layer.hpp"
#include "json/json.h"
#include <functional>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/cl.hpp"

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
        virtual void forward_gpu(const CLLayerData& prev, CLLayerData& current);
        virtual void backward_cpu(LayerData& prev, LayerData& current);
        virtual void backward_gpu(CLLayerData& prev, CLLayerData& current);

        virtual std::unique_ptr<LayerData> createLayerData(size_t train_num);

        virtual void importLayer(const Json::Value& coeffs);
        virtual Json::Value exportLayer();

        virtual std::string what() { return "sigmoid"; }
        virtual size_t getNeuronNum() const;

        void setDropout(bool enable);

        static const std::function<float(float)> f_sigmoid;
        static const std::function<float(float)> f_sigmoid_prime;

    private:
        void refreshCLLayerInfo();

        void refreshDropout();
        void updateDOBuffer();

        const size_t m_prev_d, m_current_d;
        float m_learn_rate;

        const bool m_uses_dropout;
        bool m_dropout_enabled;
        const float m_dropout_rate;
        const bool m_uses_gpu;

        float *m_weight;
        float *m_bias;
        float *m_dropout_coeff;

        // OpenCL contexts
        cl::Buffer m_buf_w, m_buf_b, m_buf_do;
        cl::Kernel m_fwd_kernel;

    public:
        virtual void setLearnRate(float rate) { m_learn_rate = rate; }
        virtual float getLearnRate() const { return m_learn_rate; }
    };
}

#endif // __SIGMOID_LAYER_HPP
