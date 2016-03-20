#include "layers/cl_layer_data.hpp"
#include "utils/cl_exception.hpp"
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

    void CLLayerData::loadToCLBuffer(DataIndex idx)
    {
        cl::CommandQueue queue = CLContext::getInstance().getCommandQueue();
        cl_int err = queue.enqueueWriteBuffer(m_buffers[static_cast<int>(idx)],
                CL_TRUE,
                0,
                sizeof(float) * getDataNum() * getTrainNum(),
                get(idx));
        printError(err, "Error at CommandQueue::enqueueWriteBuffer in CLLayerData::loadToCLBuffer");
    }

    void CLLayerData::getFromCLBuffer(DataIndex idx)
    {
        cl::CommandQueue queue = CLContext::getInstance().getCommandQueue();
        cl_int err = queue.enqueueReadBuffer(m_buffers[static_cast<int>(idx)],
                CL_TRUE,
                0,
                sizeof(float) * getDataNum() * getTrainNum(),
                get(idx));
        printError(err, "Error at CommandQueue::enqueueReadBuffer in CLLayerData::loadToCLBuffer");
    }

    cl::Buffer CLLayerData::getCLBuffer(LayerData::DataIndex idx) const
    {
        return m_buffers[static_cast<int>(idx)];
    }
}
