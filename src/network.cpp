#include <string>
#include <utility>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <queue>
#include <stack>
#include <cstring>
#include <random>
#include <cmath>
#include "network.hpp"
#include "calc/calc-cpu.hpp"
#include "calc/util-functions.hpp"
#include "layers/cl_buffer_layer_data.hpp"
#include "layers/cl_image_layer_data.hpp"
#include "layers/layer_factory.hpp"
#include "layers/sigmoid_layer.hpp"
#include "layers/layer_merger.hpp"
#include "utils/make_unique.hpp"

namespace NeuralNet
{
    struct Network::Node
    {
        std::unique_ptr<Layer> layer;
        std::unique_ptr<LayerData> data;
        std::string file_path;
        std::vector<NodeID> next_id;
        NodeID prev_id;
    };

    struct Network::MergerNode
    {
        std::unique_ptr<LayerMerger> merger;
        std::unique_ptr<LayerData> data;
        std::vector<NodeID> prev_id;
    };

    struct Network::TestSet
    {
        std::string name;
        std::vector<float> data;
        std::vector< std::vector<int> > category_list;

        // copy-construct data & category list
        explicit TestSet(const std::string& _name, const std::vector<float>& _data,
                const std::vector< std::vector<int> >& _categ_list)
            : name(_name), data(_data), category_list(_categ_list) {}

        // move-construct data & category list
        explicit TestSet(const std::string& _name, std::vector<float>&& _data,
                std::vector< std::vector<int> >&& _categ_list)
            : name(_name), data(_data), category_list(_categ_list) {}
    };

    /* may emit Json::Exception while execution */
    Network::Network(const Json::Value& setting)
    {
        SettingMapType setting_map;

        /* train size and batch size */
        m_train_size = setting["train_num"].asUInt();
        m_batch_size = setting["batch_size"].asUInt();
        m_epoch_num = setting["epoch_num"].asUInt();

        m_max_eval_patch = setting["max_eval_patch"].asUInt();

        // learn rate setting
        m_learn_rate = setting["learn_rate"].asDouble();

        auto lr_drop_value = setting["learn_rate_drop"];
        m_learn_rate_set.enable = lr_drop_value["enable"].asBool();
        if (m_learn_rate_set.enable)
        {
            m_learn_rate_set.rate = lr_drop_value["rate"].asDouble();
            m_learn_rate_set.drop_count = lr_drop_value["drop_count"].asUInt();
            m_learn_rate_set.drop_thresh = lr_drop_value["drop_thresh"].asDouble();
            m_learn_rate_set.halt_thresh_rate =
                lr_drop_value["halt_thresh_rate"].asDouble();
        }

        // additional learning setting
        m_uses_gpu = setting["uses_gpu"].asBool();
        auto wd_value = setting["weight_decay"];
        if (!wd_value.isNull())
        {
            m_weight_decay = wd_value.asFloat();
        }
        else
        {
            m_weight_decay = 0;
        }

        /* input file description */
        auto input_val = setting["input"];
        auto input_type_str = input_val["type"].asString();

        if (!input_type_str.compare("vector"))
        {
            m_in_type = InputType::VECTOR;
            m_in_dim.size = input_val["size"].asUInt();
            m_unit_size = m_in_dim.size;
        }
        else if (!input_type_str.compare("image"))
        {
            m_in_type = InputType::IMAGE;
            auto input_size = input_val["size"];
            m_in_dim.width = input_size["width"].asUInt();
            m_in_dim.height = input_size["height"].asUInt();
            m_in_dim.channel_num = input_size["channel_num"].asUInt();
            m_unit_size = m_in_dim.width * m_in_dim.height * m_in_dim.channel_num;
        }
        else
            throw Json::LogicError("invalid input.type string");

        // layer construction 
        for (auto& start_id: setting["start_id"])
        {
            m_start_idxes.push_back(start_id.asInt());
        }

        auto layers = setting["layers"];
        if (layers.size() <= 0)
            throw Json::LogicError("there must be at least one layer");

        for (auto& start_id: m_start_idxes)
        {
            if (m_in_type == InputType::VECTOR)
            {
                auto node_pair = std::make_pair(start_id,
                    std::make_unique<LayerFactory::SigmoidLayerSetting>(
                        m_in_dim.size, m_learn_rate, 1.0, false, m_uses_gpu, m_weight_decay));
                setting_map.insert(std::move(node_pair));
            }
            else if (m_in_type == InputType::IMAGE)
            {
                auto node_pair = std::make_pair(start_id,
                    std::make_unique<LayerFactory::ImageLayerSetting>(
                        m_in_dim.width, m_in_dim.height, m_in_dim.channel_num,
                        m_learn_rate));
                setting_map.insert(std::move(node_pair));
            }
        }

        for (auto& json_layer: layers)
        {
            addLayer(json_layer, setting_map);
        }

        /* finding leaf nodes and calculation of number of output nodes */
        for (auto& node_pair: node_map)
        {
            if (node_pair.second->next_id.empty())
            {
                m_leaf_idx.push_back(node_pair.first);
            }
        }
        std::sort(m_leaf_idx.begin(), m_leaf_idx.end(),
            [](const NodeID& id1, const NodeID& id2) -> bool {
                return (id1 < id2);
            }
        );
    }

    // the given setting may be "moved"
    void Network::insertLayerSetting(SettingMapType& prevSetting,
            std::unique_ptr<LayerFactory::LayerSetting>& set,
            Network::NodeID id, Network::NodeID child_id)
    {
        if (prevSetting.find(child_id) == prevSetting.end())
        {
            prevSetting[child_id] = std::move(set);
        }
        else
        {
            if (merger_map.find(child_id) == merger_map.end())
            {
                // create merge node for the common child
                merger_map[child_id] = std::make_unique<MergerNode>();
                merger_map[child_id]->merger = std::make_unique<LayerMerger>();

                for (auto& node_pair: node_map)
                {
                    for (auto& node_child: node_pair.second->next_id)
                    {
                        if (child_id == node_child)
                        {
                            merger_map[child_id]->prev_id.push_back(node_pair.first);
                            merger_map[child_id]->merger->add(node_pair.first,
                                    node_pair.second->layer->getNeuronNum());
                            break;
                        }
                    }
                }

                // update prevSetting
                prevSetting[child_id] = std::make_unique<LayerFactory::SigmoidLayerSetting>(
                        merger_map[child_id]->merger->getNeuronNum(),
                        0.01, 1, false, m_uses_gpu, m_weight_decay
                );
            }
            else
            {
                // update the merger node
                merger_map[child_id]->merger->add(id,
                        node_map[id]->layer->getNeuronNum());

                // update prevSetting
                auto& prev_set = dynamic_cast<LayerFactory::SigmoidLayerSetting&>(*(prevSetting[child_id]));
                prev_set.neuron_num += node_map[id]->layer->getNeuronNum();
            }
        }
    }

    void Network::addLayer(const Json::Value& jsonLayer, SettingMapType& prevSetting)
    {
        auto layer_type = jsonLayer["type"].asString();
        auto layer_id = jsonLayer["id"].asInt();
        auto layer_child = jsonLayer["child"];
        auto layer_data_path = jsonLayer["data_location"].asString();
        auto layer_dim = jsonLayer["dimensions"];

        node_map[layer_id] = std::make_unique<Node>();
        node_map[layer_id]->file_path = std::move(layer_data_path);
        node_map[layer_id]->prev_id = getParent(layer_id);
        for (auto& child_val: layer_child)
        {
            NodeID id_child = child_val.asInt();
            node_map[layer_id]->next_id.push_back(id_child);
        }

        std::unique_ptr<LayerFactory::LayerSetting> cur_setting;

        if (!layer_type.compare("sigmoid"))
        {
            auto layer_enable_do = jsonLayer["enable_dropout"].asBool();
            size_t neurons = layer_dim["size"].asUInt();

            float layer_do_rate = 1.0;
            if (layer_enable_do)
                layer_do_rate = jsonLayer["dropout_rate"].asDouble();

            cur_setting = std::make_unique<LayerFactory::SigmoidLayerSetting>(
                    neurons, m_learn_rate, layer_do_rate, layer_enable_do, m_uses_gpu,
                    m_weight_decay);
        }
        else if (!layer_type.compare("convolution"))
        {
            size_t maps = layer_dim["map_num"].asUInt();
            size_t recep = layer_dim["recep_size"].asUInt();
            bool zeropad = layer_dim["enable_zero_pad"].asBool();
            size_t input_w = 0, input_h = 0;

            LayerFactory::getInstance().getOutputDimension(prevSetting[layer_id].get(),
                    input_w, input_h);

            cur_setting = std::make_unique<LayerFactory::ConvLayerSetting>(maps,
                    recep, input_w, input_h, m_learn_rate, zeropad, m_uses_gpu,
                    m_weight_decay);
        }
        else if (!layer_type.compare("maxpool"))
        {
            size_t pw = layer_dim["pool_width"].asUInt();
            size_t ph = layer_dim["pool_height"].asUInt();
            size_t st = layer_dim["stride"].asUInt();
            size_t input_w = 0, input_h = 0;
            size_t map_num = LayerFactory::getInstance().getMapNum(
                    prevSetting[layer_id].get());

            LayerFactory::getInstance().getOutputDimension(prevSetting[layer_id].get(),
                    input_w, input_h);

            cur_setting = std::make_unique<LayerFactory::MaxPoolLayerSetting>(
                    map_num, pw, ph, input_w, input_h, st, m_uses_gpu);
        }
        else
            throw Json::LogicError("invalid layer type");

        node_map[layer_id]->layer = LayerFactory::getInstance().makeLayer(
                prevSetting[layer_id].get(), cur_setting.get());

        for (auto& id: node_map[layer_id]->next_id)
        {
            insertLayerSetting(prevSetting, cur_setting, layer_id, id);
        }
    }

    /* returns itself if the given node ID indicates the root node
     * returns -1 if an invalid id is given
     */
    Network::NodeID Network::getParent(const Network::NodeID& id) const
    {
        for (auto& start_id: m_start_idxes)
        {
            if (id == start_id)
                return id;
        }

        for (const auto& node_pair: node_map)
        {
            const auto& next_ids = node_pair.second->next_id;
            if (std::find(next_ids.begin(), next_ids.end(), id) != next_ids.end())
                return node_pair.first;
        }

        return -1;
    }

    Network::~Network()
    {
    }

    void Network::loadFromFiles()
    {
        for (auto& node_pair: node_map)
        {
            auto node_id = node_pair.first;
            auto& node = *(node_pair.second);
            if (node.file_path.empty())
                continue;

            Json::CharReaderBuilder builder;
            builder["collectComments"] = false;

            Json::Value dataValue;
            std::string errors;
            std::fstream dataStream(node.file_path, std::ios_base::in);
            bool ok = Json::parseFromStream(builder, dataStream, &dataValue, &errors);
            if (ok)
            {
                auto data_id = dataValue["id"].asInt();
                auto data_type = dataValue["type"].asString();

                /* TODO: data type check? */

                if (data_id == node_id)
                {
                    node.layer->importLayer(dataValue["data"]);
                }
                else
                    throw Json::LogicError("error loading file at loadFromFiles()");
            }
            else
                std::cerr << "Error at Network::loadFromFiles(): "
                    << errors << std::endl;
        }
    }

    void Network::storeIntoFiles()
    {
        for (auto& node_pair: node_map)
        {
            auto node_id = node_pair.first;
            auto& node = *(node_pair.second);
            if (node.file_path.empty())
                continue;

            Json::Value dataValue(Json::objectValue);
            dataValue["id"] = Json::Value(node_id);
            dataValue["type"] = Json::Value(node.layer->what());
            dataValue["data"] = node.layer->exportLayer();

            Json::StyledStreamWriter writer("    ");
            std::fstream dataStream(node.file_path, std::ios_base::out);
            writer.write(dataStream, dataValue);
        }
    }

    void Network::registerTestSet(const std::string& name, const std::vector<float>& data,
            const std::vector< std::vector<int> >& categ_list)
    {
        m_list_testset.emplace_back(name, data, categ_list);
    }

    void Network::registerTestSet(const std::string& name, std::vector<float>&& data,
            std::vector< std::vector<int> >&& categ_list)
    {
        m_list_testset.emplace_back(name, data, categ_list);
    }

    // used for evaluation of a number of data
    std::vector< std::vector<int> > Network::evaluateAll(const std::vector<float>& data)
    {
        if (data.size() % m_unit_size != 0)
        {
            throw NetworkException("evaluate(): size of data does not match with that "
                    "of the network");
        }
        size_t num_data = data.size() / m_unit_size;

        // disable dropout of sigmoid layers for evaluation
        for (auto& node_pair: node_map)
        {
            auto* layer_ptr = node_pair.second->layer.get();
            auto* sigmoid_ptr = dynamic_cast<SigmoidLayer *>(layer_ptr);
            if (sigmoid_ptr)
            {
                sigmoid_ptr->setDropout(false);
            }
        }

        std::vector< std::vector<int> > retval(m_leaf_idx.size());
        for (size_t idx_d = 0; idx_d < num_data; idx_d += m_max_eval_patch)
        {
            size_t test_data_num = std::min(m_max_eval_patch, num_data - idx_d);

            std::vector<size_t> lst_idxes;
            for (size_t i = idx_d; i < idx_d + test_data_num; i++)
            {
                lst_idxes.push_back(i);
            }

            prepareLayerData(test_data_num);
            feedForward(data, lst_idxes);

            // return value generation
            for (size_t i = 0; i < m_leaf_idx.size(); i++)
            {
                auto* cl_data_ptr = dynamic_cast<CLLayerData *>(node_map[m_leaf_idx[i]]->data.get());
                if (cl_data_ptr)
                {
                    cl_data_ptr->getFromCL(LayerData::DataIndex::ACTIVATION);
                }

                auto category_list = getCategory(*(node_map[m_leaf_idx[i]]->data));
                retval[i].insert(retval[i].end(), category_list.begin(),
                        category_list.end());
            }
        }

        return retval;
    }

    // evaluate for a single data
    std::vector<int> Network::evaluate(const std::vector<float>& data)
    {
        if (data.size() != m_unit_size)
            throw NetworkException("evaluate(): size of data does not match with that "
                    "of the network");

        auto retval = evaluateAll(data);
        std::vector<int> oned_retval;
        for (auto& value_list: retval)
        {
            oned_retval.push_back(value_list[0]);
        }
        return oned_retval;
    }

    void Network::evaluateTestSets()
    {
        for (auto& test_set: m_list_testset)
        {
            testTestSet(test_set);
        }
    }

    void Network::feedForward(const std::vector<float>& data,
            const std::vector<size_t>& list_idx)
    {
        /* fill the data in the input LayerData */
        for (size_t i=0; i<list_idx.size(); i++)
        {
            auto data_idx = list_idx[i];
            copy_vec(data.data() + (data_idx * m_unit_size),
                    m_input_data->get(LayerData::DataIndex::ACTIVATION) + i * m_unit_size,
                    m_unit_size);
        }

        auto* cl_data_ptr = dynamic_cast<CLLayerData *>(m_input_data.get());
        if (cl_data_ptr)
        {
            cl_data_ptr->loadToCL(LayerData::DataIndex::ACTIVATION);
        }

        // forward the input
        for (auto& node_pair: node_map)
        {
            feedForwardLayer(node_pair.first);
        }
    }

    std::vector<int> Network::getCategory(const LayerData& data) const
    {
        std::vector<int> retval;

        for (size_t t = 0; t < data.getTrainNum(); t++)
        {
            const float* out = data.get(LayerData::DataIndex::ACTIVATION)
                    + (t * data.getDataNum());
            int maxi=0;

            for (size_t i=1; i<data.getDataNum(); i++)
            {
                if (out[i] > out[maxi])
                    maxi = i;
            }

            retval.push_back(maxi);
        }

        return retval;
    }

    void Network::backPropagate()
    {
        /* error value initialization of non-output layers at the beginning of the calculation */
        for (auto& node_pair: node_map)
        {
            if (std::find(m_leaf_idx.begin(), m_leaf_idx.end(), node_pair.first)
                    == m_leaf_idx.end())
            {
                auto& node_data = *(node_pair.second->data);
                memset(node_data.get(LayerData::DataIndex::ERROR), 0,
                        sizeof(node_data.getDataNum() * node_data.getTrainNum() * sizeof(float)));
            }
        }

        // traverse the nodes for back propagation
        for (auto it_pair = node_map.rbegin(); it_pair != node_map.rend(); it_pair++)
        {
            backPropagateLayer(it_pair->first);
        }
    }

    void Network::prepareLayerData(size_t train_num)
    {
        if (m_input_data && train_num == m_input_data->getTrainNum())
            return;

        if (m_uses_gpu)
        {
            if (m_in_type == InputType::VECTOR)
            {
                m_input_data = std::make_unique<CLBufferLayerData>(
                        train_num, m_unit_size);
            }
            else
            {
                m_input_data = std::make_unique<CLImageLayerData>(
                        train_num,
                        m_in_dim.width, m_in_dim.height, m_in_dim.channel_num,
                        CLImageLayerData::Channel::INTENSITY);
            }
        }
        else
        {
            m_input_data = std::make_unique<LayerData>(train_num, m_unit_size);
        }

        for (auto& node_pair: node_map)
        {
            node_pair.second->data =
                    std::move(node_pair.second->layer->createLayerData(train_num));
        }
        for (auto& node_pair: merger_map)
        {
            node_pair.second->data =
                    std::move(node_pair.second->merger->createLayerData(train_num));
        }
    }

    // returns list of error values for each test case
    void Network::calcOutputErrors(
            const std::vector< std::vector<int> >& category_list,
            const std::vector< size_t >& batch_idxes,
            std::vector<float>& error_vals,
            size_t batch_num)
    {
        for (size_t i = 0; i < m_leaf_idx.size(); i++)
        {
            auto& leaf_id = m_leaf_idx[i];
            auto& leaf_node = *(node_map[leaf_id]);
            auto output_nodes = leaf_node.data->getDataNum();

            std::vector<float> sprime_z(m_batch_size * output_nodes, 0);
            copy_vec(leaf_node.data->get(LayerData::DataIndex::INTER_VALUE),
                    sprime_z.data(),
                    output_nodes * m_batch_size);
            apply_vec(sprime_z.data(), sprime_z.data(), output_nodes * m_batch_size,
                    ActivationFuncs::f_sigmoid_prime);

            std::vector<float> deriv_cost(m_batch_size * output_nodes, 0);
            for (size_t j = 0; j < m_batch_size; j++)
            {
                for (size_t k = 0; k < output_nodes; k++)
                {
                    deriv_cost[j * output_nodes + k] =
                            -category_list[i][output_nodes * batch_idxes[j] + k];
                }
            }

            // softmax output value calculation
            std::vector<float> softmax_output(m_batch_size * output_nodes, 0);
            apply_vec(leaf_node.data->get(LayerData::DataIndex::ACTIVATION),
                    softmax_output.data(), output_nodes * m_batch_size,
                    [](float in) -> float {
                        return std::exp(in);
                    });
            for (size_t j = 0; j < m_batch_size; j++)
            {
                float expsum = 0;
                for (size_t k = 0; k < output_nodes; k++)
                {
                    expsum += softmax_output[j*output_nodes + k];
                }
                for (size_t k = 0; k < output_nodes; k++)
                {
                    softmax_output[j*output_nodes + k] /= expsum;
                }
            }

            // error value calculation
            std::vector<float> batch_errors(m_batch_size * output_nodes, 0);
            apply_vec(softmax_output.data(), batch_errors.data(),
                    output_nodes * m_batch_size,
                    [](float in) -> float {
                        return std::log(in);
                    });
            pmul_vec(batch_errors.data(), deriv_cost.data(), batch_errors.data(),
                    output_nodes * m_batch_size);
            for (size_t j = 0; j < m_batch_size; j++)
            {
                float errorsum = 0;
                for (size_t k = 0; k < output_nodes; k++)
                {
                    errorsum += batch_errors[j*output_nodes + k];
                }
                error_vals[batch_num*m_batch_size + j] += errorsum;
            }

            add_vec(softmax_output.data(),
                    deriv_cost.data(),
                    deriv_cost.data(),
                    output_nodes * m_batch_size);

            pmul_vec(sprime_z.data(), deriv_cost.data(),
                    leaf_node.data->get(LayerData::DataIndex::ERROR),
                    output_nodes * m_batch_size);
        }
    }

    void Network::dropLearnRate(const std::vector<float>& total_errors)
    {
        if (!(m_learn_rate_set.enable))
            return;

        const int window_first_idx = total_errors.size() -
                m_learn_rate_set.drop_count - 1;
        if (window_first_idx < 0)
            return;

        bool error_not_falling = true;
        for (size_t i = window_first_idx + 1; i < total_errors.size(); i++)
        {
            if (total_errors[window_first_idx] - total_errors[i] >= m_learn_rate_set.drop_thresh)
            {
                error_not_falling = false;
            }
        }
        if (!error_not_falling)
            return;

        m_learn_rate *= m_learn_rate_set.rate;
        for (auto& node_pair: node_map)
        {
            node_pair.second->layer->setLearnRate(m_learn_rate);
        }

        std::cout << "learn rate dropped to: " << m_learn_rate << std::endl;
    }

    void Network::train(const std::vector<float>& data,
            const std::vector< std::vector<int> >& category_list)
    {
        std::vector<size_t> data_idxes;
        for (size_t i=0; i<m_train_size; i++)
            data_idxes.push_back(i);

        std::vector<size_t> batch_idxes(m_batch_size);

        std::random_device rd;
        std::mt19937 rgen(rd());

        std::vector<float> total_errors;

        for (size_t epoch = 0; epoch < m_epoch_num; epoch++)
        {
            prepareLayerData(m_batch_size);
            std::shuffle(data_idxes.begin(), data_idxes.end(), rgen);

            // enable dropout of sigmoid layers for training
            for (auto& node_pair: node_map)
            {
                auto* layer_ptr = node_pair.second->layer.get();
                auto* sigmoid_ptr = dynamic_cast<SigmoidLayer *>(layer_ptr);
                if (sigmoid_ptr)
                {
                    sigmoid_ptr->setDropout(true);
                }
            }

            std::vector<float> error_vals(m_train_size, 0);

            for (size_t batch_num = 0; batch_num * m_batch_size < m_train_size; batch_num++)
            {
                for (size_t i = 0; i < m_batch_size; i++)
                {
                    batch_idxes[i] = data_idxes[i + batch_num * m_batch_size];
                }

                feedForward(data, batch_idxes);

                for (auto& node_pair: node_map)
                {
                    auto* cl_layer_data = dynamic_cast<CLLayerData *>(node_pair.second->data.get());
                    if (cl_layer_data)
                    {
                        cl_layer_data->getFromCL(LayerData::DataIndex::ACTIVATION);
                        cl_layer_data->getFromCL(LayerData::DataIndex::INTER_VALUE);
                    }
                }

                calcOutputErrors(category_list, batch_idxes, error_vals, batch_num);
                backPropagate();
            }

            std::cout << "epoch #" << (epoch+1) <<
                " finished, saving data..." << std::endl;
            storeIntoFiles();

            // print the total error value for this epoch
            float total_error = 0;
            for (auto& error: error_vals)
            {
                total_error += error;
            }
            std::cout << "total error = " << total_error << std::endl;

            total_errors.push_back(total_error);
            
            evaluateTestSets();

            dropLearnRate(total_errors);
            if (m_learn_rate <= m_learn_rate_set.halt_thresh_rate)
            {
                std::cout << "learn rate is below the threshold (" << 
                    m_learn_rate_set.halt_thresh_rate << "). Stop." << std::endl;
                break;
            }
        }
    }

    void Network::testTestSet(const Network::TestSet& testset)
    {
        std::cout << "Testing test " << testset.name << ":";

        auto result = evaluateAll(testset.data);
        if (result.size() != testset.category_list.size())
            throw NetworkException("invalid size of given category lists");
        for (size_t i = 0; i < result.size(); i++)
        {
            if (result[i].size() != testset.category_list[i].size())
                throw NetworkException("invalid size of given category lists");

            size_t count = 0;
            for (size_t j = 0; j < result[i].size(); j++)
            {
                if (result[i][j] == testset.category_list[i][j])
                    count++;
            }

            std::cout << " (" << count << "/" << result[i].size() << ")";
        }

        std::cout << std::endl;
    }

    std::vector<Network::NodeID> Network::feedForwardLayer(const Network::NodeID& in_idx)
    {
        auto& in_node = *(node_map[in_idx]);

        if (std::find(m_start_idxes.begin(), m_start_idxes.end(), in_idx)
                != m_start_idxes.end())
        {
            // the node is an input node
            in_node.layer->forward(*(m_input_data), *(in_node.data), m_uses_gpu);
        }
        else if (merger_map.find(in_idx) != merger_map.end())
        {
            // the node is a merging node
            auto& merge_node = *(merger_map[in_idx]);
            for (auto& p_idx: merge_node.prev_id)
            {
                merge_node.merger->assign(p_idx, *(node_map[p_idx]->data),
                        *(merge_node.data));
            }
            in_node.layer->forward(*(merge_node.data), *(in_node.data), m_uses_gpu);
        }
        else
        {
            auto& prev_node = *(node_map[in_node.prev_id]);
            in_node.layer->forward(*(prev_node.data), *(in_node.data), m_uses_gpu);
        }

        return in_node.next_id;
    }

    const Network::NodeID Network::backPropagateLayer(const Network::NodeID& in_idx)
    {
        auto& in_node = *(node_map[in_idx]);

        if (std::find(m_start_idxes.begin(), m_start_idxes.end(), in_idx)
                != m_start_idxes.end())
        {
            // the node is an input node
            in_node.layer->backward(*(m_input_data), *(in_node.data), m_uses_gpu);
        }
        else if (merger_map.find(in_idx) != merger_map.end())
        {
            // the node is a merging node
            auto& merge_node = *(merger_map[in_idx]);
            in_node.layer->backward(*(merge_node.data), *(in_node.data), m_uses_gpu);

            std::map< NodeID, LayerData* > parent_datas;
            for (auto& p_idx: merge_node.prev_id)
            {
                parent_datas[p_idx] = (node_map[p_idx]->data).get();
            }
            merge_node.merger->distribute(parent_datas, *(merge_node.data));
        }
        else
        {
            auto& prev_node = *(node_map[in_node.prev_id]);
            in_node.layer->backward(*(prev_node.data), *(in_node.data), m_uses_gpu);
        }

        return in_node.prev_id;
    }
}
