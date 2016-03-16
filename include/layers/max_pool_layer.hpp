#ifndef __MAX_POOL_LAYER_HPP
#define __MAX_POOL_LAYER_HPP

#include "layers/layer_data.hpp"
#include "layers/layer.hpp"
#include "json/json.h"
#include <cstdlib>

namespace NeuralNet
{
    class MaxPoolLayer: public Layer
    {
    public:
        struct Dimension
        {
            size_t map_num;
            size_t image_width;
            size_t image_height;
            size_t pool_width;
            size_t pool_height;
            size_t stride;
        };

        MaxPoolLayer(const Dimension& dim);
        virtual ~MaxPoolLayer() {}

        virtual void forward_cpu(const LayerData& prev, LayerData& current);
        virtual void forward_gpu(const LayerData& prev, LayerData& current);
        virtual void backward_cpu(LayerData& prev, LayerData& current);
        virtual void backward_gpu(LayerData& prev, LayerData& current);

        virtual std::unique_ptr<LayerData> createLayerData(size_t train_num);

        virtual void importLayer(const Json::Value& coeffs);
        virtual Json::Value exportLayer();

        virtual std::string what() { return "maxpool"; }
        virtual size_t getNeuronNum() const;

        virtual void setLearnRate(float rate) {}
        virtual float getLearnRate() const { return 0; }

    private:
        const Dimension m_dim;
        const size_t m_output_width, m_output_height;
    };
}

#endif
