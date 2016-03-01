#ifndef __CONV_LAYER_HPP
#define __CONV_LAYER_HPP

#include "layers/layer.hpp"
#include "layers/layer_data.hpp"
#include "json/json.h"
#include <cstdlib>
#include <memory>

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
            double learn_rate;
            bool enable_zero_pad;
        };

        enum class ActivationFunc
        {
            SIGMOID, RELU,
        };

        ConvLayer(const LayerSetting& setting, ActivationFunc func);
        virtual ~ConvLayer();

        virtual void forward_cpu(const LayerData& prev, LayerData& current);
        virtual void forward_gpu(const LayerData& prev, LayerData& current);
        virtual void backward_cpu(LayerData& prev, LayerData& current);
        virtual void backward_gpu(LayerData& prev, LayerData& current);

        virtual std::unique_ptr<LayerData> createLayerData(size_t train_num);

        virtual void importLayer(const Json::Value& coeffs);
        virtual Json::Value exportLayer();

        virtual std::string what() { return "convolution"; }

    private:
        LayerSetting m_set;
        const double m_learn_rate;
        size_t m_output_width, m_output_height;

        std::function<double(double)> f_activation;
        std::function<double(double)> f_activation_prime;
        void (*f_convolution)(const double *, const double *, double *,
                int, int, int, int);
        void (*f_convol_back)(const double *, const double *, double *,
                int, int, int, int);

        double *m_weight;
    };
}

#endif // __CONV_LAYER_HPP
