#include "layers/layer_data.hpp"

namespace NeuralNet
{
	LayerData::LayerData(size_t train_num, size_t prev_len, size_t current_len)
		: m_train_num(train_num), m_prev_len(prev_len), m_current_len(current_len)
	{
		/* memory allocation */
		data = new double*[DATA_COUNT];
		data[static_cast<int>(DataIndex::ACTIVATION)] = new double[train_num * current_len];
		data[static_cast<int>(DataIndex::INTER_VALUE)] = new double[train_num * current_len];
		data[static_cast<int>(DataIndex::WEIGHT)] = new double[prev_len * current_len];
		data[static_cast<int>(DataIndex::BIAS)] = new double[current_len];
		data[static_cast<int>(DataIndex::ERROR)] = new double[train_num * current_len];
	}
	
	LayerData::~LayerData()
	{
		for (int i = static_cast<int>(DataIndex::START); i <= static_cast<int>(DataIndex::END);
				i++)
		{
			delete [] data[i];
		}
		delete [] data;
	}
	
	void LayerData::resize(size_t train_num, size_t prev_len, size_t current_len)
	{
		/* TODO */
	}
	
	double *LayerData::get(LayerData::DataIndex idx) const
	{
		return data[static_cast<size_t>(idx)];
	}
}