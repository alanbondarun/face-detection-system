#ifndef __LAYER_DATA_HPP
#define __LAYER_DATA_HPP

#include <cstdlib>

namespace NeuralNet
{
    class LayerData
    {
    public:
        enum class DataIndex
        {
            ACTIVATION,
            INTER_VALUE,
            ERROR,
            START = ACTIVATION,
            END = ERROR
        };

        static constexpr size_t DATA_COUNT = static_cast<int>(DataIndex::END)
                - static_cast<int>(DataIndex::START) + 1;

        LayerData(size_t train_num, size_t data_num);
        virtual ~LayerData();

        LayerData(const LayerData& other);
        LayerData& operator=(const LayerData& other);

        /* returns the desired array */
        float *get(DataIndex idx) const;

        /* returns the dimensions */
        size_t getDataNum() const { return m_data_num; }
        size_t getTrainNum() const { return m_train_num; }

    private:
        size_t m_train_num, m_data_num;
        float *data;
    };
}

#endif // __LAYER_DATA_HPP
