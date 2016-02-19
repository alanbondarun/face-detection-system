#ifndef __NETWORK_HPP
#define __NETWORK_HPP

#include <memory>
#include <string>
#include <utility>
#include <cstdlib>
#include <unordered_map>
#include <map>
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
		
	public:
		using NodeUPtr = std::unique_ptr<Node>;
		using NodeID = std::pair<int, int>;
	
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
		
		// returns classification value for one portion of data
		std::vector<int> evaluate(const std::vector<double>& data);
		std::vector< std::vector<int> > evaluate(const std::vector<double>& data,
				const std::vector<size_t>& list_idx);
		
		// trains with m_train_size number of data
		void train(const std::vector<double>& data, const std::vector<int>& category_list);
		
	private:
		void addLayer(const Json::Value& jsonLayer,
				std::map< NodeID, LayerFactory::LayerSetting >& prevSetting);
				
		std::vector<NodeID> feedForward(const NodeID& in_idx);
		const NodeID backPropagate(const NodeID& in_idx);
		
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
		
		NodeID root_idx;
		std::vector<NodeID> m_leaf_idx;
		std::map< NodeID, NodeUPtr > node_map;
		
		std::unique_ptr<LayerData> m_input_data;
	};
}

#endif // __NETWORK_HPP