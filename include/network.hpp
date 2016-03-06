#ifndef __NETWORK_HPP
#define __NETWORK_HPP

#include <memory>
#include <string>
#include <utility>
#include <cstdlib>
#include <unordered_map>
#include <map>
#include <unordered_set>
#include "json/json.h"
#include "layers/layer.hpp"
#include "layers/layer_data.hpp"
#include "layers/layer_factory.hpp"

namespace NeuralNet
{
    class Network
    {
    public:
        /* enum for input type */
        enum class InputType
        {
            VECTOR, IMAGE
        };

    private:
        struct Node;
//        struct MergerNode;

        using NodeUPtr = std::unique_ptr<Node>;
//        using MergerNodeUPtr = std::unique_ptr<MergerNode>;

    public:
        /* type aliases */
        using NodeID = int;
        using SettingMapType = std::map<
                NodeID,
                std::pair<
                    LayerFactory::LayerType,
                    std::unique_ptr<LayerFactory::LayerSetting>
                >
        >;

        /* an exception thrown when this class received invalid JSON data */
        class InvalidJSONException
        {
        public:
            InvalidJSONException(const std::string& region)
                : m_reg(std::string("Invalid JSON setting") + region) {}
            virtual ~InvalidJSONException() {}
            virtual const char* what() const { return m_reg.c_str(); }

        private:
            const std::string m_reg;
        };

        Network(const Json::Value& setting);
        ~Network();

        /* load/store neuron coefficients from files */
        void loadFromFiles();
        void storeIntoFiles();

        // returns classification value for one portion of data
        std::vector< int > evaluate(const std::vector<double>& data);

        // trains with m_train_size number of data
        // category_list: list of (list of desired output data for each input data)
        //    for each output layer
        void train(const std::vector<double>& data,
            const std::vector< std::vector<int> >& category_list);

    private:
        // parse and add layer(s) from one JSON layer block
        // more than one layer may be add if the JSON block is a branch
        void addLayer(const Json::Value& jsonLayer, SettingMapType& prevSetting);

        /* helper function for propagation */
        void feedForward(const std::vector<double>& data,
                const std::vector<size_t>& list_idx);
        void backPropagate();

        /* helper function for propagation of one layer */
        std::vector<NodeID> feedForwardLayer(const NodeID& in_idx);
        const NodeID backPropagateLayer(const NodeID& in_idx);

        NodeID getParent(const NodeID& id) const;

        std::vector<int> getCategory(const LayerData& data) const;

        /* dimensions */
        InputType m_in_type;
        struct InputSize
        {
            size_t size;
            size_t width, height, channel_num;
        } m_in_dim;
        size_t m_unit_size, m_train_size, m_batch_size, m_epoch_num;
        double m_learn_rate;

        std::vector<NodeID> m_start_idxes;
        std::vector<NodeID> m_leaf_idx;
        std::map< NodeID, NodeUPtr > node_map;
//        std::map< NodeID, MergerNodeUPtr> merger_map;

        std::unique_ptr<LayerData> m_input_data;
    };
}

#endif // __NETWORK_HPP
