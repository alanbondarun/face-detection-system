#include <string>
#include "network.hpp"
#include "utils/make_unique.hpp"

namespace NeuralNet
{
	struct Network::Node
	{
		std::unique_ptr<Layer> layer;
		std::unique_ptr<LayerData> data;
		std::vector<Node *> next;
		Node *prev;
	};
	
	/* may emit Json::Exception while execution */
	Network::Network(const Json::Value& setting)
		: root(nullptr)
	{
		/* input file description */
		auto input_val = setting["input"];
		auto input_type_str = input_val["type"].asString();
		if (!input_type_str.compare("vector"))
		{
			m_in_type = InputType::VECTOR;
			m_in_size.size = input_val["size"].asUInt();
		}
		else if (!input_type_str.compare("image"))
		{
			m_in_type = InputType::IMAGE;
			auto input_size = input_val["size"];
			m_in_size.width = input_size["width"].asUInt();
			m_in_size.height = input_size["height"].asUInt();
		}
		else
			throw Json::LogicError("invalid input.type string");
		
		/* train size and batch size */
		m_train_size = setting["train_num"].asUInt();
		m_batch_size = setting["batch_size"].asUInt();
		
		/* layer construction */
		auto layers = setting["layers"];
		if (layers.size() <= 0)
			throw Json::LogicError("there must be at least one layer");

		auto layer1 = layers[0u];
		addLayer(layer1);
		for (size_t idx = 1; idx < layers.size(); idx++)
		{
			addLayer(layers[idx]);
		}
	}
	
	void Network::addLayer(const Json::Value& jsonLayer)
	{
		
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
	
	std::vector<Network::Node *> Network::feedForward(Network::Node *in)
	{
		
	}
	
	Network::Node* Network::backPropagate(Network::Node *in)
	{
		
	}
}