#ifndef __LAYER_MERGER_HPP
#define __LAYER_MERGER_HPP

#include <vector>
#include <functional>
#include <utility>
#include <map>
#include <memory>
#include "layers/layer_data.hpp"

namespace NeuralNet
{
    class LayerMerger
    {
    public:
        using KeyType = int;

        LayerMerger();
        LayerMerger(const std::vector< std::pair<KeyType, size_t> >& parentNodes);

        void add(KeyType key, size_t num_neuron);

        void assign(const KeyType& key, const LayerData& parent_data,
                LayerData& this_data);

        void distribute(std::map< KeyType, LayerData* >& parent_datas,
                const LayerData& this_data);

        std::unique_ptr<LayerData> createLayerData(size_t train_num);

    private:
        size_t m_neuron_num;
        std::map< KeyType, size_t > m_cumul_idxes;
        std::map< KeyType, size_t > m_parent_sizes;

    public:
        size_t getNeuronNum() const { return m_neuron_num; }
    };
}

#endif // __LAYER_MERGER_HPP
