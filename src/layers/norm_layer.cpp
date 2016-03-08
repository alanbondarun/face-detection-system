#include "layers/norm_layer.hpp"
#include "calc/calc-cpu.hpp"
#include "utils/make_unique.hpp"

namespace NeuralNet
{
    NormalizeLayer::NormalizeLayer(const Setting& set)
        : m_set(set),
        m_output_neurons(m_set.map_num * m_set.input_width * m_set.input_height)
    {
    }

    NormalizeLayer::~NormalizeLayer()
    {
    }

    void NormalizeLayer::forward_cpu(const LayerData& prev, LayerData& current)
    {
        auto train_num = current.getTrainNum();
        auto prev_z = prev.get(LayerData::DataIndex::INTER_VALUE);
        auto prev_a = prev.get(LayerData::DataIndex::ACTIVAITON);
        auto current_z = current.get(LayerData::DataIndex::INTER_VALUE);
        auto current_a = current.get(LayerData::DataIndex::ACTIVAITON);

        copy_vec(prev_z, current_z, train_num * m_output_neurons);

        size_t a_offset = 0;
        for (size_t i = 0; i < train_num; i++)
        {
            for (size_t j = 0; j < m_set.map_num; j++)
            {
                lr_normalize_mat(prev_a + a_offset, cur_a + a_offset,
                        m_set.input_width, m_set.input_height,
                        m_set.halfwidth, m_set.alpha, m_set.beta);
                a_offset += (m_set.input_width * m_set.input_height);
            }
        }
    }

    void NormalizeLayer::forward_gpu(const LayerData& prev, LayerData& current)
    {
        /* TODO: OpenCL intergration */
    }

    void NormalizeLayer::backward_cpu(LayerData& prev, LayerData& current)
    {
        // TODO!!!
    }

    void NormalizeLayer::backward_gpu(LayerData& prev, LayerData& current)
    {
        /* TODO: OpenCL intergration */
    }

    std::unique_ptr<LayerData> NormalizeLayer::createLayerData(size_t train_num)
    {
        return std::make_unique<LayerData>(
                train_num,
                m_output_neurons
        );
    }

    void NormalizeLayer::importLayer(const Json::Value& coeffs)
    {
    }

    Json::Value NormalizeLayer::exportLayer()
    {
        return Json::Value();
    }

    size_t NormalizeLayer::getNeuronNum() const
    {
        return m_output_neurons;
    }
}
