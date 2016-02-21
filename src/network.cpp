#include <string>
#include <utility>
#include <algorithm>
#include <queue>
#include <stack>
#include <cstring>
#include <random>
#include "network.hpp"
#include "calc/calc-cpu.hpp"
#include "calc/util-functions.hpp"
#include "layers/layer_factory.hpp"
#include "utils/make_unique.hpp"

namespace NeuralNet
{
	struct Network::Node
	{
		std::unique_ptr<Layer> layer;
		std::unique_ptr<LayerData> data;
		std::vector<NodeID> next_id;
		NodeID prev_id;
	};
	
	/* may emit Json::Exception while execution */
	Network::Network(const Json::Value& setting)
	{
		/* train size and batch size */
		m_train_size = setting["train_num"].asUInt();
		m_batch_size = setting["batch_size"].asUInt();
		m_epoch_num = setting["epoch_num"].asUInt();
		
		m_learn_rate = setting["learn_rate"].asDouble();
		
		/* input file description */
		auto input_val = setting["input"];
		auto input_type_str = input_val["type"].asString();
		LayerFactory::LayerSetting first_setting;
		
		if (!input_type_str.compare("vector"))
		{
			m_in_type = InputType::VECTOR;
			m_in_dim.size = input_val["size"].asUInt();
			m_unit_size = m_in_dim.size;
			
			first_setting = LayerFactory::SigmoidLayerSetting(m_train_size,
					m_in_dim.size, m_learn_rate);
		}
		else if (!input_type_str.compare("image"))
		{
			m_in_type = InputType::IMAGE;
			auto input_size = input_val["size"];
			m_in_dim.width = input_size["width"].asUInt();
			m_in_dim.height = input_size["height"].asUInt();
			m_in_dim.channel_num = input_size["channel_num"].asUInt();
			m_unit_size = m_in_dim.width * m_in_dim.height * m_in_dim.channel_num;
			
			first_setting = LayerFactory::ImageLayerSetting(m_train_size,
					m_in_dim.width, m_in_dim.height, m_in_dim.channel_num);
		}
		else
			throw Json::LogicError("invalid input.type string");
		
		/* layer construction */
		auto layers = setting["layers"];
		if (layers.size() <= 0)
			throw Json::LogicError("there must be at least one layer");

		std::map< NodeID, LayerFactory::LayerSetting > setting_map;		

		auto layer1 = layers[0u];
		root_idx = std::make_pair(layer1["id"].asInt(), 0);
		setting_map[root_idx] = first_setting;
		addLayer(layer1, setting_map);
		
		for (int idx = 1; idx < layers.size(); idx++)
		{
			addLayer(layers[idx], setting_map);
		}
		
		/* input layer data construction */
		m_input_data = std::make_unique<LayerData>(m_batch_size, m_unit_size);
		
		/* finding leaf nodes and calculation of number of output nodes */
		m_output_size = 0;
		for (auto& node_pair: node_map)
		{
			if (node_pair.second->next_id.empty())
			{
				m_leaf_idx.push_back(node_pair.first);
				m_output_size += node_pair.second->data->getDataNum();
			}
		}
	}
	
	void Network::addLayer(const Json::Value& jsonLayer,
			std::map< Network::NodeID, LayerFactory::LayerSetting >& prevSetting)
	{
		auto layer_id = jsonLayer["id"].asInt();
		auto layer_type = jsonLayer["type"].asString();
		auto layer_children = jsonLayer["children"];
		
		std::vector<NodeID> vec_child;
		for (const auto& child_val: layer_children)
			vec_child.push_back(std::make_pair(child_val.asInt(), 0));
		
		if (!layer_type.compare("branch"))
		{
			auto layer_sizes = jsonLayer["sizes"];
			std::vector<int> vec_sizes;
			for (const auto& size_val: layer_sizes)
				vec_sizes.push_back(size_val.asInt());
			
			for (int i=0; i<vec_sizes.size(); i++)
			{
				auto layer_setting = LayerFactory::SigmoidLayerSetting(m_batch_size, vec_sizes[i], m_learn_rate);
				auto idx = std::make_pair(layer_id, i);
				auto first_idx = std::make_pair(layer_id, 0);
				
				std::unique_ptr<Layer> layer = LayerFactory::makeLayer(
						LayerFactory::LayerType::SIGMOID, prevSetting[first_idx], layer_setting);
						
				node_map[idx] = std::make_unique<Node>();
				node_map[idx]->data = std::move(layer->createLayerData());
				node_map[idx]->layer = std::move(layer);
				node_map[idx]->next_id.push_back(vec_child[i]);
				
				auto parent_idx = getParent(first_idx);
				if (i > 0 && first_idx != root_idx)
					node_map[parent_idx]->next_id.push_back(idx);
				node_map[idx]->prev_id = parent_idx;
				
				prevSetting[idx] = layer_setting;
			}
		}
		else /* not branch layer */
		{
			LayerFactory::LayerSetting cur_setting;
			if (!layer_type.compare("sigmoid"))
			{
				size_t neurons = jsonLayer["size"].asUInt();
				cur_setting = LayerFactory::SigmoidLayerSetting(m_batch_size, neurons, m_learn_rate);
			}
			else if (!layer_type.compare("convolution"))
			{
				size_t maps = jsonLayer["map_num"].asUInt();
				size_t recep = jsonLayer["recep_size"].asUInt();
				bool zeropad = jsonLayer["enable_zero_pad"].asBool();
				cur_setting = LayerFactory::ConvLayerSetting(m_batch_size, maps, recep, m_learn_rate, zeropad);
			}
			else if (!layer_type.compare("maxpool"))
			{
				size_t pw = jsonLayer["pool_width"].asUInt();
				size_t ph = jsonLayer["pool_height"].asUInt();
				cur_setting = LayerFactory::MaxPoolLayerSetting(m_batch_size, pw, ph);
			}
			else
				throw Json::LogicError("invalid layer type");
				
			auto idx = std::make_pair(layer_id, 0);
			std::unique_ptr<Layer> layer = LayerFactory::makeLayer(
					LayerFactory::LayerType::SIGMOID, prevSetting[idx], cur_setting);
			node_map[idx] = std::make_unique<Node>();
			node_map[idx]->data = std::move(layer->createLayerData());
			node_map[idx]->layer = std::move(layer);
			node_map[idx]->next_id = std::move(vec_child);
			
			auto parent_idx = getParent(idx);
			node_map[idx]->prev_id = parent_idx;
			
			prevSetting[idx] = cur_setting;
		}
	}
	
	/* returns itself if the given node ID indicates the root node
	 * returns (-1, 0) if an invalid id is given
	 */
	Network::NodeID Network::getParent(const Network::NodeID& id) const
	{
		if (id == root_idx)
			return id;
		
		for (const auto& node_pair: node_map)
		{
			const auto& next_ids = node_pair.second->next_id;
			if (std::find(next_ids.begin(), next_ids.end(), id) != next_ids.end())
				return node_pair.first;
		}
		
		return std::make_pair(-1, 0);
	}
	
	Network::~Network()
	{
	}

	/* used for evaluation of a single set of data */
	std::vector< std::vector<int> > Network::evaluate(const std::vector<double>& data)
	{
		std::vector<size_t> lst_idxes;
		size_t idx_num = data.size() / m_unit_size;
		for (size_t i=0; i<idx_num; i++)
			lst_idxes.push_back(i);

		return evaluate(data, lst_idxes);
	}
	
	std::vector< std::vector<int> > Network::evaluate(const std::vector<double>& data,
			const std::vector<size_t>& list_idx)
	{
		feedForward(data, list_idx);
		
		/* return value generation */
		std::vector< std::vector<int> > retval;
		for (auto& leaf_id: m_leaf_idx)
		{
			retval.push_back(getCategory(*(node_map[leaf_id]->data)));
		}
		
		return retval;
	}
	
	void Network::feedForward(const std::vector<double>& data,
			const std::vector<size_t>& list_idx)
	{
		/* fill the data in the input LayerData */
		for (size_t i=0; i<list_idx.size(); i++)
		{
			auto data_idx = list_idx[i];
			copy_vec(data.data() + data_idx * m_unit_size * sizeof(double),
					m_input_data->get(LayerData::DataIndex::ACTIVATION) + i * m_unit_size * sizeof(double),
					m_unit_size * sizeof(double));
		}

		/* forward the input */
		std::queue<NodeID> nodeQueue;
		nodeQueue.push(root_idx);
		while (!nodeQueue.empty())
		{
			auto cur_id = nodeQueue.front();
			nodeQueue.pop();
			
			auto list_next_id = feedForwardLayer(cur_id);
			for (auto next_id: list_next_id)
			{
				nodeQueue.push(next_id);
			}
		}
	}
	
	std::vector<int> Network::getCategory(const LayerData& data) const
	{
		std::vector<int> retval;
		
		for (size_t t = 0; t < data.getTrainNum(); t++)
		{
			const double* out = data.get(LayerData::DataIndex::ACTIVATION)
					+ t * data.getDataNum() * sizeof(double);
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
						sizeof(node_data.getDataNum() * node_data.getTrainNum() * sizeof(double)));
			}
		}
		
		/* traverse the nodes for back propagation */
		std::map< NodeID, bool > node_traversed;
		for (auto& node_pair: node_map)
			node_traversed[node_pair.first] = false;
		
		std::stack<NodeID> node_stack;
		node_stack.push(root_idx);
		while (!node_stack.empty())
		{
			auto& top_id = node_stack.top();
			if (node_traversed[top_id])
			{
				backPropagateLayer(top_id);
				node_stack.pop();
			}
			else
			{
				node_traversed[top_id] = true;
				for (auto& child_id: node_map[top_id]->next_id)
				{
					node_stack.push(child_id);
				}
			}
		}
	}
	
	void Network::train(const std::vector<double>& data,
			const std::vector< std::vector<int> >& category_list)
	{
		std::vector<size_t> data_idxes;
		for (size_t i=0; i<m_train_size; i++)
			data_idxes.push_back(i);
		
		std::vector<size_t> batch_idxes(m_batch_size);
		
		std::random_device rd;
		std::mt19937 rgen(rd());
		
		for (size_t epoch = 0; epoch < m_epoch_num; epoch++)
		{
			std::shuffle(data_idxes.begin(), data_idxes.end(), rgen);
			
			for (size_t batch_num = 0; batch_num * m_batch_size < m_train_size; batch_num++)
			{
				for (size_t i = 0; i < m_batch_size; i++)
				{
					batch_idxes[i] = data_idxes[i + batch_num * m_batch_size];
				}
				
				feedForward(data, batch_idxes);
				
				/* error value for output layers */
				for (size_t i = 0; i < m_leaf_idx.size(); i++)
				{
					auto& leaf_id = m_leaf_idx[i];
					auto& leaf_node = *(node_map[leaf_id]);
					auto output_nodes = leaf_node.data->getDataNum();

					std::vector<double> sprime_z(m_batch_size * output_nodes, 0);
					copy_vec(leaf_node.data->get(LayerData::DataIndex::INTER_VALUE),
							sprime_z.data(),
							output_nodes * m_batch_size);
					apply_vec(sprime_z.data(), sprime_z.data(), output_nodes * m_batch_size,
							f_sigmoid_prime);
					
					std::vector<double> deriv_cost(m_batch_size * output_nodes, 0);
					for (size_t j = 0; j < m_batch_size; j++)
					{
						for (size_t k = 0; k < output_nodes; k++)
						{
							deriv_cost[j * output_nodes + k] =
									-category_list[i][output_nodes * batch_idxes[j] + k];
						}
					}
					add_vec(leaf_node.data->get(LayerData::DataIndex::ACTIVATION),
							deriv_cost.data(),
							deriv_cost.data(),
							output_nodes * m_batch_size);

					pmul_vec(sprime_z.data(), deriv_cost.data(),
							leaf_node.data->get(LayerData::DataIndex::ERROR),
							output_nodes * m_batch_size);
				}
				
				backPropagate();
			}
		}
	}
	
	std::vector<Network::NodeID> Network::feedForwardLayer(const Network::NodeID& in_idx)
	{
		auto& in_node = *(node_map[in_idx]);
		
		if (in_idx == root_idx)
		{
			in_node.layer->forward(*(m_input_data), *(in_node.data));
		}
		else
		{
			auto& prev_node = *(node_map[in_node.prev_id]);
			in_node.layer->forward(*(prev_node.data), *(in_node.data));
		}
		
		return in_node.next_id;
	}
	
	const Network::NodeID Network::backPropagateLayer(const Network::NodeID& in_idx)
	{
		auto& in_node = *(node_map[in_idx]);
		
		if (in_idx != root_idx)
		{
			auto& prev_node = *(node_map[in_node.prev_id]);
			in_node.layer->backward(*(prev_node.data), *(in_node.data));
		}
		
		return in_node.prev_id;
	}
}