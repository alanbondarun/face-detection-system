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
        cl_int err;
        std::vector<cl::Event> writeEvents;

        for (size_t i = 0; i < getTrainNum(); i++)
        {
            cl::Event ev;
            err = queue.enqueueWriteBuffer(
                    m_buffers[static_cast<int>(idx) * getTrainNum() + i],
                    CL_FALSE,
                    0,
                    sizeof(float) * getDataNum(),
                    get(idx) + (getDataNum() * i),
                    nullptr, &ev);
            printError(err, "Error at CommandQueue::enqueueWriteBuffer in "
                    "CLBufferLayerData::loadToCL");
            writeEvents.push_back(std::move(ev));
        }
        
        err = queue.enqueueMarkerWithWaitList(&writeEvents);
        printError(err, "Error at CommandQueue::enqueueMarkerWithWaitList in "
                "CLBufferLayerData::loadToCL");
    }

    void CLBufferLayerData::getFromCL(DataIndex idx)
    {
        auto queue = CLContext::getInstance().getCommandQueue();
        cl_int err;
        std::vector<cl::Event> readEvents;

        for (size_t i = 0; i < getTrainNum(); i++)
        {
            cl_bool block_read = (getDataNum() < 128)?CL_TRUE:CL_FALSE;

            cl::Event ev;
            err = queue.enqueueReadBuffer(
                    m_buffers[static_cast<int>(idx) * getTrainNum() + i],
                    block_read,
                    0,
                    sizeof(float) * getDataNum(),
                    get(idx) + (getDataNum() * i),
                    nullptr, &ev);
            printError(err, "Error at CommandQueue::enqueueReadBuffer in "
                    "CLBufferLayerData::getFromCL");
            readEvents.push_back(std::move(ev));
        }
        
        err = queue.enqueueMarkerWithWaitList(&readEvents);
        printError(err, "Error at CommandQueue::enqueueMarkerWithWaitList in "
                "CLImageLayerData::getFromCL");
    }

    cl::Memory CLBufferLayerData::getCLMemory(LayerData::DataIndex data_idx,
            size_t train_idx) const
    {
        return m_buffers[static_cast<int>(data_idx) * getTrainNum()
                + train_idx];
    }
}
