#include "layers/layer_data.hpp"
#include <utility>

namespace NeuralNet
{
    LayerData::LayerData(size_t train_num, size_t data_num)
        : m_capacity(train_num),
        m_train_num(train_num), m_data_num(data_num)
    {
        data = new float[DATA_COUNT * m_capacity * data_num];
    }

    LayerData::~LayerData()
    {
        delete [] data;
    }

    LayerData::LayerData(const LayerData& other)
        : m_capacity(other.m_capacity), m_train_num(other.m_train_num),
        m_data_num(other.m_data_num)
    {
        const size_t data_size = DATA_COUNT * m_capacity * m_data_num;

        data = new float[data_size];
        for (size_t i = 0; i < data_size; i++)
        {
            data[i] = other.data[i];
        }
    }

    LayerData& LayerData::operator=(const LayerData& other)
    {
        LayerData tmp(other);
        *this = std::move(tmp);
        return *this;
    }

    LayerData::LayerData(LayerData&& other) noexcept
        : m_capacity(other.m_capacity), m_train_num(other.m_train_num),
        m_data_num(other.m_data_num), data(other.data)
    {
        other.data = nullptr;
    }

    LayerData& LayerData::operator=(LayerData&& other) noexcept
    {
        delete [] data;

        m_capacity = other.m_capacity;
        m_train_num = other.m_train_num;
        m_data_num = other.m_data_num;
        data = other.data;
        other.data = nullptr;

        return *this;
    }

    float *LayerData::get(LayerData::DataIndex idx) const
    {
        return data + (static_cast<int>(idx) * m_capacity * m_data_num);
    }

    void LayerData::setTrainNum(size_t _t)
    {
        if (_t > m_capacity)
        {
            updateDataSize(_t);
            m_capacity = _t;
        }
        m_train_num = _t;
    }

    void LayerData::updateDataSize(size_t new_train_num)
    {
        float* tmp_data = new float[new_train_num * m_data_num * DATA_COUNT];
        for (size_t i = 0; i < DATA_COUNT; i++)
        {
            size_t new_block_offset = i * new_train_num * m_data_num;
            size_t old_block_offset = i * m_capacity * m_data_num;
            for (size_t j = 0; j < m_train_num * m_data_num; j++)
                tmp_data[j + new_block_offset] = data[j + old_block_offset];
        }
        data = tmp_data;
    }
}
