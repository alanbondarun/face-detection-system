#include "layers/layer_data.hpp"

namespace NeuralNet
{
	LayerData::LayerData(size_t train_num, size_t data_num)
		: m_train_num(train_num), m_data_num(data_num)
	{
		/* memory allocation */
		data = new double[DATA_COUNT * train_num * data_num];
	}
	
	LayerData::~LayerData()
	{
		delete [] data;
	}
	
	void LayerData::resize(size_t train_num, size_t data_num)
	{
		/* TODO */
	}
	
	double *LayerData::get(LayerData::DataIndex idx) const
	{
		return data + (static_cast<int>(idx) * m_train_num * m_data_num);
	}
}