#include "layers/cl_layer_data.hpp"
#include "cl_context.hpp"

namespace NeuralNet
{
    CLLayerData::CLLayerData(size_t train_num, size_t data_num)
        : LayerData(train_num, data_num)
    {
        cl::Context context = CLContext::getInstance().getContext();
        for (size_t i = 0; i < LayerData::DATA_COUNT; i++)
        {
            m_buffers.emplace_back(context, CL_MEM_READ_WRITE,
                    sizeof(float) * train_num * data_num);
        }
    }

    CLLayerData::~CLLayerData()
    {
    }

    cl::Buffer CLLayerData::getCLBuffer(LayerData::DataIndex idx) const
    {
        return m_buffers[static_cast<int>(idx)];
    }
}
