#ifndef __LAYER_FACTORY_HPP
#define __LAYER_FACTORY_HPP

#include <utility>
#include <vector>
#include <memory>
#include "layers/layer.hpp"

namespace NeuralNet
{
    /* early declaration */
    class LayerCreatorBase;
    template <typename T1, typename T2> class LayerCreator;

    class LayerFactory
    {
    public:
        enum class LayerType
        {
            NONE, SIGMOID, CONVOLUTION, MAXPOOL, IMAGE
        };

        /* layer setting structs */
        struct LayerSetting
        {
            virtual ~LayerSetting() {}
        };
        struct SigmoidLayerSetting: public LayerSetting
        {
            size_t neuron_num;
            float learn_rate;
            float dropout_rate;
            bool enable_dropout;
            bool uses_gpu;
            explicit SigmoidLayerSetting(size_t _n, float _l, float _droprate, bool _dr, bool _gpu)
                : LayerSetting(), neuron_num(_n), learn_rate(_l), dropout_rate(_droprate),
                enable_dropout(_dr), uses_gpu(_gpu) {}
            virtual ~SigmoidLayerSetting() {}
        };
        struct ImageLayerSetting: public LayerSetting
        {
            size_t image_w, image_h, channel_num;
            float learn_rate;
            explicit ImageLayerSetting(size_t _w, size_t _h, size_t _c, float _lr)
                : LayerSetting(), image_w(_w), image_h(_h),
                channel_num(_c), learn_rate(_lr) {}
            virtual ~ImageLayerSetting() {}
        };
        struct ConvLayerSetting: public LayerSetting
        {
            size_t map_num, recep_size, input_w, input_h;
            float learn_rate;
            bool enable_zero_pad, uses_gpu;
            size_t output_w, output_h;
            explicit ConvLayerSetting(size_t _m, size_t _r, size_t _iw, size_t _ih, float _l, bool _zeropad, bool _gpu)
                : LayerSetting(), map_num(_m), recep_size(_r), input_w(_iw), input_h(_ih),
                learn_rate(_l), enable_zero_pad(_zeropad), uses_gpu(_gpu),
                output_w((enable_zero_pad)?(_iw):(_iw - (_r - 1))),
                output_h((enable_zero_pad)?(_ih):(_ih - (_r - 1))) {}
            virtual ~ConvLayerSetting() {}
        };
        struct MaxPoolLayerSetting: public LayerSetting
        {
            size_t map_num, pool_w, pool_h, input_w, input_h, output_w, output_h, stride;
            bool uses_gpu;
            explicit MaxPoolLayerSetting(size_t _m, size_t _w, size_t _h, size_t _iw, size_t _ih, size_t _st, bool _gpu)
                : LayerSetting(), map_num(_m), pool_w(_w), pool_h(_h), input_w(_iw), input_h(_ih),
                output_w(_iw / _w), output_h(_ih / _h), stride(_st), uses_gpu(_gpu) {}
            virtual ~MaxPoolLayerSetting() {}
        };

        /* allow the usage of ctor only within this class */
    private:
        LayerFactory();

    public:
        LayerFactory(const LayerFactory&) = delete;
        LayerFactory& operator=(const LayerFactory&) = delete;
        LayerFactory(LayerFactory&&) = delete;
        LayerFactory& operator=(const LayerFactory&&) = delete;

        static LayerFactory& getInstance()
        {
            static LayerFactory in;
            return in;
        }

        /* default factory function.
         * returns empty unique_ptr for invalid layer type
         */
        std::unique_ptr<Layer> makeLayer(const LayerSetting* prev_setting,
                const LayerSetting* cur_setting);

        /* helper functions */
        void getOutputDimension(const LayerSetting* set,
                size_t& width, size_t& height);
        size_t getMapNum(const LayerSetting* set);

        LayerType whatType(const LayerSetting* set);

    private:
        std::map< std::pair<LayerType, LayerType>,
                std::unique_ptr<LayerCreatorBase> > m_creators;
    };

    class LayerCreatorBase
    {
    public:
        LayerCreatorBase() {}
        virtual ~LayerCreatorBase() {}
        virtual std::unique_ptr<Layer> create(const LayerFactory::LayerSetting& prev_set,
                const LayerFactory::LayerSetting& cur_set) = 0;
    };

    template <typename T1, typename T2>
    class LayerCreator : public LayerCreatorBase
    {
    public:
        LayerCreator() {}
        virtual ~LayerCreator() {}
        virtual std::unique_ptr<Layer> create(const LayerFactory::LayerSetting& prev_set,
                const LayerFactory::LayerSetting& cur_set)
        {
            return std::unique_ptr<Layer>();
        }
    };
}

#endif // __LAYER_FACTORY_HPP
