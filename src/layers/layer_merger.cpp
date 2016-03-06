#include "layers/layer_merger.hpp"
#include "utils/make_unique.hpp"
#include "calc/calc-cpu.hpp"

namespace NeuralNet
{
    LayerMerger::LayerMerger()
        : m_neuron_num(0)
    {
    }

    LayerMerger::LayerMerger(const std::vector< std::pair<KeyType, size_t> >& parentNodes)
    {
        size_t cumul_num = 0;
        for (auto& node: parentNodes)
        {
            m_cumul_idxes[node.first] = cumul_num;
            m_parent_sizes[node.first] = node.second;
            cumul_num += node.second;
        }
        m_neuron_num = cumul_num;
    }

    void LayerMerger::add(KeyType key, size_t num_neuron)
    {
        if (m_cumul_idxes.find(key) == m_cumul_idxes.end())
        {
            m_cumul_idxes[key] = m_neuron_num;
            m_parent_sizes[key] = num_neuron;
            m_neuron_num += num_neuron;
        }
    }

    void LayerMerger::assign(const LayerMerger::KeyType& key, const LayerData& parent_data,
            LayerData& this_data)
    {
        const auto offset = m_cumul_idxes[key];
        const auto data_d = m_parent_sizes[key];
        auto parent_a = parent_data.get(LayerData::DataIndex::ACTIVATION);
        auto parent_z = parent_data.get(LayerData::DataIndex::INTER_VALUE);
        auto this_a = this_data.get(LayerData::DataIndex::ACTIVATION);
        auto this_z = this_data.get(LayerData::DataIndex::INTER_VALUE);

        copy_vec(parent_a, this_a + offset, data_d);
        copy_vec(parent_z, this_z + offset, data_d);
    }

    void LayerMerger::distribute(
            std::map< KeyType, LayerData* >& parent_datas,
            const LayerData& this_data)
    {
        auto this_e = this_data.get(LayerData::DataIndex::ERROR);

        for (auto& data_pair: parent_datas)
        {
            auto offset = m_cumul_idxes[data_pair.first];
            auto data_d = m_parent_sizes[data_pair.first];
            auto parent_e = data_pair.second->get(LayerData::DataIndex::ERROR);

            copy_vec(this_e + offset, parent_e, data_d);
        }
    }

    std::unique_ptr<LayerData> LayerMerger::createLayerData(size_t train_num)
    {
        return std::make_unique<LayerData>(
                train_num, m_neuron_num
        );
    }
}
