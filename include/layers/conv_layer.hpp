#ifndef __CONV_LAYER_HPP
#define __CONV_LAYER_HPP

#include "layers/layer.hpp"
#include "layers/layer_data.hpp"
#include "json/json.h"
#include <cstdlib>
#include <memory>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

namespace NeuralNet
{
    class ConvLayer: public Layer
    {
    public:
        struct LayerSetting
        {
            size_t prev_map_num;
            size_t current_map_num;
            size_t image_width;
            size_t image_height;
            size_t recep_size;
            float learn_rate;
            bool enable_zero_pad;
            bool uses_gpu;
        };

        enum class ActivationFunc
        {
            SIGMOID, RELU,
        };

        ConvLayer(const LayerSetting& setting, ActivationFunc func);
        virtual ~ConvLayer();

        virtual void forward_cpu(const LayerData& prev, LayerData& current);
        virtual void forward_gpu(const CLLayerData& prev, CLLayerData& current);
        virtual void backward_cpu(LayerData& prev, LayerData& current);
        virtual void backward_gpu(CLLayerData& prev, CLLayerData& current);

        virtual std::unique_ptr<LayerData> createLayerData(size_t train_num);

        virtual void importLayer(const Json::Value& coeffs);
        virtual Json::Value exportLayer();

        virtual std::string what() { return "convolution"; }
        virtual size_t getNeuronNum() const;

    private:
        void refreshCLLayerInfo();

        LayerSetting m_set;
        float m_learn_rate;
        size_t m_output_width, m_output_height;

        std::function<float(float)> f_activation;
        std::function<float(float)> f_activation_prime;
        void (*f_convolution)(const float *, const float *, float *,
                int, int, int, int);
        void (*f_convol_back)(const float *, const float *, float *,
                int, int, int, int);

        float *m_weight;
        float *m_bias;

        cl::Buffer m_buf_w, m_buf_b;
        cl::Kernel m_fwd_kernel;

    public:
        virtual void setLearnRate(float rate) { m_learn_rate = rate; }
        virtual float getLearnRate() const { return m_learn_rate; }
    };
}

#endif // __CONV_LAYER_HPP
