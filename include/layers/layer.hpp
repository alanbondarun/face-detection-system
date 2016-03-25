#ifndef __LAYER_HPP
#define __LAYER_HPP

#include <cstdlib>
#include <memory>
#include <string>
#include "layers/layer_data.hpp"
#include "layers/cl_layer_data.hpp"
#include "json/json.h"

namespace NeuralNet
{
    /**
     * interface representing a layer.
     */
    class Layer
    {
    public:
        virtual ~Layer() {}

        /* forwarding input of the layer */
        void forward(const LayerData& prev, LayerData& current, bool uses_gpu)
        {
            if (uses_gpu)
            {
                forward_gpu(dynamic_cast<const CLLayerData&>(prev),
                        dynamic_cast<CLLayerData&>(current));
            }
            else
                forward_cpu(prev, current);
        }

        /* backpropagation of the layer */
        void backward(LayerData& prev, LayerData& current, bool uses_gpu)
        {
            if (uses_gpu)
            {
                backward_gpu(dynamic_cast<CLLayerData&>(prev),
                        dynamic_cast<CLLayerData&>(current));
            }
            else
                backward_cpu(prev, current);
        }

        /* creation of appropriate layer data for the layer */
        virtual std::unique_ptr<LayerData> createLayerData(size_t train_num) = 0;

        /* import/export of layer coefficients.
         * importLayer() may emit Json::Exception during execution
         */
        virtual void importLayer(const Json::Value& coeffs) = 0;
        virtual Json::Value exportLayer() = 0;

        /* description of the layer */
        virtual std::string what() = 0;
        virtual size_t getNeuronNum() const = 0;

        // learn rate modulation
        virtual void setLearnRate(float rate) = 0;
        virtual float getLearnRate() const = 0;

    protected:
        /**
         * CPU and GPU versions of the forward() and backward() that child classes
         * need to implement
         */
        virtual void forward_cpu(const LayerData& prev, LayerData& current) = 0;
        virtual void forward_gpu(const CLLayerData& prev, CLLayerData& current) = 0;
        virtual void backward_cpu(LayerData& prev, LayerData& current) = 0;
        virtual void backward_gpu(CLLayerData& prev, CLLayerData& current) = 0;
    };
}

#endif // __LAYER_HPP
