#ifndef __NETWORK_HPP
#define __NETWORK_HPP

#include <memory>
#include <string>
#include <cstdlib>
#include <unordered_map>
#include "json/json.h"
#include "layers/layer.hpp"
#include "layers/layer_data.hpp"

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
		std::vector<int> evaluate(const double *data) const;
		
		// trains with m_train_size number of data
		void train(const double *data);
		
	private:
		struct Node;
		using NodeUPtr = std::unique_ptr<Node>;
		
		void addLayer(const Json::Value& jsonLayer);
		std::vector<int> feedForward(int in_idx);
		int backPropagate(int in_idx);
		
		InputType m_in_type;
		struct InputSize
		{
			size_t size;
			size_t width, height;
		} m_in_size;
		size_t m_train_size, m_batch_size;
		
		int root_idx;
		std::unordered_map<int, NodeUPtr> node_map;
	};
}

#endif // __NETWORK_HPP