#include <string>
#include <utility>
#include <algorithm>
#include "network.hpp"
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
					m_in_dim.size);
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
		inputData = std::make_unique<LayerData>(m_batch_size, m_unit_size);
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
				auto layer_setting = LayerFactory::SigmoidLayerSetting(m_batch_size, vec_sizes[i]);
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
				cur_setting = LayerFactory::SigmoidLayerSetting(m_batch_size, neurons);
			}
			else if (!layer_type.compare("convolution"))
			{
				size_t maps = jsonLayer["map_num"].asUInt();
				size_t recep = jsonLayer["recep_size"].asUInt();
				bool zeropad = jsonLayer["enable_zero_pad"].asBool();
				cur_setting = LayerFactory::ConvLayerSetting(m_batch_size, maps, recep, zeropad);
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

	std::vector<int> Network::evaluate(const std::vector<double>& data) const
	{
		return evaluate(data, 0);
	}
	
	std::vector<int> Network::evaluate(const std::vector<double>& data, size_t data_idx) const
	{
		
	}
	
	void Network::train(const std::vector<double>& data, const std::vector<int>& category_list)
	{
		
	}
	
	std::vector<int> Network::feedForward(int in_idx)
	{
		
	}
	
	int Network::backPropagate(int in_idx)
	{
		
	}
}