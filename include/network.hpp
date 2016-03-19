#ifndef __NETWORK_HPP
#define __NETWORK_HPP

#include <memory>
#include <exception>
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
        struct MergerNode;
        struct TestSet;

        struct LearnRateSetting
        {
            bool enable;
            float rate;
            size_t drop_count;
            float drop_thresh;
            float halt_thresh_rate;
        };

        using NodeUPtr = std::unique_ptr<Node>;
        using MergerNodeUPtr = std::unique_ptr<MergerNode>;

    public:
        /* type aliases */
        using NodeID = int;
        using SettingMapType = std::map<
                NodeID,
                std::unique_ptr<LayerFactory::LayerSetting>
        >;

        /* an exception thrown when this class received invalid JSON data */
        class NetworkException: public std::exception
        {
        public:
            NetworkException(const std::string& msg)
                : m_msg(std::string("Error at Network: ") + msg) {}
            virtual ~NetworkException() {}
            virtual const char* what() const noexcept { return m_msg.c_str(); }

        private:
            const std::string m_msg;
        };

        Network(const Json::Value& setting);
        ~Network();

        /* load/store neuron coefficients from files */
        void loadFromFiles();
        void storeIntoFiles();

        // insert test data to the network (copy-construct)
        void registerTestSet(const std::string& name, const std::vector<float>& data,
                const std::vector< std::vector<int> >& categ_list);

        // insert test data to the network (move-construct)
        void registerTestSet(const std::string& name, std::vector<float>&& data,
                std::vector< std::vector<int> >&& categ_list);

        // returns classification value for one portion of data
        std::vector< int > evaluate(const std::vector<float>& data);

        // returns classification values for a set of data
        std::vector< std::vector<int> > evaluateAll(const std::vector<float>& data);

        // trains with m_train_size number of data
        // category_list: list of (list of desired output data for each input data)
        //    for each output layer
        void train(const std::vector<float>& data,
            const std::vector< std::vector<int> >& category_list);

    private:
        void insertLayerSetting(SettingMapType& prevSetting,
                std::unique_ptr<LayerFactory::LayerSetting>& set,
                NodeID id, NodeID child_id);

        void prepareLayerData(size_t train_num);

        void testTestSet(const TestSet& testset);

        // parse and add layer(s) from one JSON layer block
        // more than one layer may be add if the JSON block is a branch
        void addLayer(const Json::Value& jsonLayer, SettingMapType& prevSetting);

        /* helper function for propagation */
        void feedForward(const std::vector<float>& data,
                const std::vector<size_t>& list_idx);
        void backPropagate();

        /* helper function for propagation of one layer */
        std::vector<NodeID> feedForwardLayer(const NodeID& in_idx);
        const NodeID backPropagateLayer(const NodeID& in_idx);

        NodeID getParent(const NodeID& id) const;

        std::vector<int> getCategory(const LayerData& data) const;

        void calcOutputErrors(
                const std::vector< std::vector<int> >& category_list,
                const std::vector< size_t >& batch_idxes,
                std::vector<float>& error_vals,
                size_t batch_num);

        void dropLearnRate(const std::vector<float>& total_errors);

        /* dimensions */
        InputType m_in_type;
        struct InputSize
        {
            size_t size;
            size_t width, height, channel_num;
        } m_in_dim;
        size_t m_unit_size, m_train_size, m_batch_size, m_epoch_num;
        size_t m_max_eval_patch;
        float m_learn_rate;
        LearnRateSetting m_learn_rate_set;
        bool m_uses_gpu;

        std::vector<NodeID> m_start_idxes;
        std::vector<NodeID> m_leaf_idx;
        std::map< NodeID, NodeUPtr > node_map;
        std::map< NodeID, MergerNodeUPtr> merger_map;

        std::unique_ptr<LayerData> m_input_data;
        std::vector< TestSet > m_list_testset;
    };
}

#endif // __NETWORK_HPP
