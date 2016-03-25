#include "layers/cl_buffer_layer_data.hpp"
#include "utils/cl_exception.hpp"
#include "cl_context.hpp"

namespace NeuralNet
{
    CLBufferLayerData::CLBufferLayerData(size_t train_num, size_t data_num)
        : CLLayerData(train_num, data_num)
    {
        auto context = CLContext::getInstance().getContext();
        for (size_t i = 0; i < LayerData::DATA_COUNT * train_num; i++)
        {
            m_buffers.emplace_back(context, CL_MEM_READ_WRITE,
                    sizeof(float) * data_num);
        }
    }

    CLBufferLayerData::~CLBufferLayerData()
    {
    }

    void CLBufferLayerData::loadToCL(DataIndex idx)
    {
        auto queue = CLContext::getInstance().getCommandQueue();
        for (size_t i = 0; i < getTrainNum(); i++)
        {
            cl_int err = queue.enqueueWriteBuffer(
                    m_buffers[static_cast<int>(idx) * getTrainNum() + i],
                    CL_TRUE,
                    0,
                    sizeof(float) * getDataNum(),
                    get(idx) + (getDataNum() * i));
            printError(err, "Error at CommandQueue::enqueueWriteBuffer in "
                    "CLBufferLayerData::loadToCL");
        }
    }

    void CLBufferLayerData::getFromCL(DataIndex idx)
    {
        auto queue = CLContext::getInstance().getCommandQueue();
        for (size_t i = 0; i < getTrainNum(); i++)
        {
            cl_int err = queue.enqueueReadBuffer(
                    m_buffers[static_cast<int>(idx) * getTrainNum() + i],
                    CL_TRUE,
                    0,
                    sizeof(float) * getDataNum(),
                    get(idx) + (getDataNum() * i));
            printError(err, "Error at CommandQueue::enqueueReadBuffer in "
                    "CLBufferLayerData::getFromCL");
        }
    }

    cl::Memory CLBufferLayerData::getCLMemory(LayerData::DataIndex data_idx,
            size_t train_idx) const
    {
        return m_buffers[static_cast<int>(data_idx) * getTrainNum()
                + train_idx];
    }
}
