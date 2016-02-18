#include <string>
#include <utility>
#include "network.hpp"
#include "layers/layer_factory.hpp"
#include "utils/make_unique.hpp"

namespace NeuralNet
{
	struct Network::Node
	{
		std::unique_ptr<Layer> layer;
		std::unique_ptr<LayerData> data;
		std::vector<int> next_id;
		int prev_id;
	};
	
	/* may emit Json::Exception while execution */
	Network::Network(const Json::Value& setting)
		: root_idx(0)
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
			m_in_size.size = input_val["size"].asUInt();
			
			first_setting = LayerFactory::SigmoidLayerSetting(m_train_size,
					m_in_size.size);
		}
		else if (!input_type_str.compare("image"))
		{
			m_in_type = InputType::IMAGE;
			auto input_size = input_val["size"];
			m_in_size.width = input_size["width"].asUInt();
			m_in_size.height = input_size["height"].asUInt();
			m_in_size.channel_num = input_size["channel_num"].asUInt();
			
			first_setting = LayerFactory::ImageLayerSetting(m_train_size,
					m_in_size.width, m_in_size.height, m_in_size.channel_num);
		}
		else
			throw Json::LogicError("invalid input.type string");
		
		/* layer construction */
		auto layers = setting["layers"];
		if (layers.size() <= 0)
			throw Json::LogicError("there must be at least one layer");

		auto layer1 = layers[0u];
		auto layer_setting = addLayer(layer1, first_setting);
		for (int idx = 1; idx < layers.size(); idx++)
		{
			layer_setting = addLayer(layers[idx], layer_setting);
		}
		
		/* previous node id verification: TODO */
	}
	
	LayerFactory::LayerSetting Network::addLayer(const Json::Value& jsonLayer,
			const LayerFactory::LayerSetting& prev_setting)
	{
		auto layer_id = jsonLayer["id"].asInt();
		auto layer_type = jsonLayer["type"].asString();
		auto layer_children = jsonLayer["children"];
		
		std::vector<int> vec_child;
		for (const auto& child_val: layer_children)
			vec_child.push_back(child_val.asInt());
		
		LayerFactory::LayerSetting cur_setting;
		
		if (!layer_type.compare("branch"))
		{
			/* TODO / early return maybe... */
		}
		else if (!layer_type.compare("sigmoid"))
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
				
		std::unique_ptr<Layer> layer = LayerFactory::makeLayer(
				LayerFactory::LayerType::SIGMOID, prev_setting, cur_setting);
		
		node_map[layer_id] = std::make_unique<Node>();
		node_map[layer_id]->data = std::move(layer->createLayerData());
		node_map[layer_id]->layer = std::move(layer);
		node_map[layer_id]->next_id = std::move(vec_child);
		
		return cur_setting;
	}
	
	Network::~Network()
	{
		
	}
	
	std::vector<int> Network::evaluate(const double *data) const
	{
	}
	
	void Network::train(const double *data)
	{
		
	}
	
	std::vector<int> Network::feedForward(int in_idx)
	{
		
	}
	
	int Network::backPropagate(int in_idx)
	{
		
	}
}