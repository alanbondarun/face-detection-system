#include "layers/cl_buffer_layer_data.hpp"
#include "utils/cl_exception.hpp"
#include "cl_context.hpp"

namespace NeuralNet
{
    CLBufferLayerData::CLBufferLayerData(size_t train_num, size_t data_num)
        : CLLayerData(train_num, data_num)
    {
        cl::ImageFormat imgfmt{CL_INTENSITY, CL_FLOAT};
        auto context = CLContext::getInstance().getContext();

        for (size_t i = 0; i < LayerData::DATA_COUNT; i++)
        {
            m_imgbufs.emplace_back(context, CL_MEM_READ_WRITE,
                    imgfmt, data_num, train_num);
        }

        m_origin[0] = m_origin[1] = m_origin[2] = 0;
        m_region[0] = data_num;
        m_region[1] = train_num;
        m_region[2] = 1;
    }

    CLBufferLayerData::~CLBufferLayerData()
    {
    }

    void CLBufferLayerData::loadToCL(DataIndex idx)
    {
        auto queue = CLContext::getInstance().getCommandQueue();
        cl_int err;

        err = queue.enqueueWriteImage(
                m_imgbufs[static_cast<int>(idx)],
                CL_TRUE,
                m_origin, m_region, 0, 0,
                get(idx));
        printError(err, "Error at CommandQueue::enqueueWriteBuffer in "
                "CLBufferLayerData::loadToCL");
    }

    void CLBufferLayerData::getFromCL(DataIndex idx)
    {
        auto queue = CLContext::getInstance().getCommandQueue();
        cl_int err;

        err = queue.enqueueReadImage(
                m_imgbufs[static_cast<int>(idx)],
                CL_TRUE,
                m_origin, m_region, 0, 0,
                get(idx));
        printError(err, "Error at CommandQueue::enqueueReadBuffer in "
                "CLBufferLayerData::getFromCL");
    }

    cl::Memory CLBufferLayerData::getCLMemory(LayerData::DataIndex data_idx) const
    {
        return m_imgbufs[static_cast<int>(data_idx)];
    }
}
