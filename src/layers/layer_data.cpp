#include "layers/layer_data.hpp"

namespace NeuralNet
{
	LayerData::LayerData(size_t prev_len, size_t current_len)
	{
		m_prev_len = prev_len;
		m_current_len = current_len;
		
		/* memory allocation */
		data = new double*[DATA_COUNT];
		data[ACTIVATION] = new double[current_len];
		data[INTER_VALUE] = new double[current_len];
		data[WEIGHT] = new double[prev_len * current_len];
		data[BIAS] = new double[current_len];
		data[ERROR] = new double[current_len];
	}
	
	LayerData::~LayerData()
	{
		for (int i = static_cast<int>(START); i <= static_cast<int>(END);
				i++)
		{
			delete [] data[i];
		}
		delete [] data;
	}
	
	void LayerData::resize(size_t prev_len, size_t current_len)
	{
		/* TODO */
	}
	
	double *LayerData::get(LayerData::DataIndex idx) const
	{
		return data[static_cast<size_t>(idx)];
	}
}