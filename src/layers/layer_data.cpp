#include "layers/layer_data.hpp"

namespace NeuralNet
{
    LayerData::LayerData(size_t train_num, size_t data_num)
        : m_train_num(train_num), m_data_num(data_num)
    {
        /* memory allocation */
        data = new float[DATA_COUNT * train_num * data_num];
    }

    LayerData::~LayerData()
    {
        delete [] data;
    }

    LayerData::LayerData(const LayerData& other)
        : m_train_num(other.m_train_num), m_data_num(other.m_data_num)
    {
        const size_t data_size = DATA_COUNT * other.m_train_num * other.m_data_num;
        data = new float[data_size];
        for (size_t i = 0; i < data_size; i++)
        {
            data[i] = other.data[i];
        }
    }

    LayerData& LayerData::operator=(const LayerData& other)
    {
        delete [] data;

        const size_t data_size = DATA_COUNT * other.m_train_num * other.m_data_num;
        data = new float[data_size];
        for (size_t i = 0; i < data_size; i++)
        {
            data[i] = other.data[i];
        }
        return *this;
    }

    float *LayerData::get(LayerData::DataIndex idx) const
    {
        return data + (static_cast<int>(idx) * m_train_num * m_data_num);
    }
}
