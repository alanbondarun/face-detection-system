#include "layers/cl_image_layer_data.hpp"
#include "utils/cl_exception.hpp"
#include "cl_context.hpp"

namespace NeuralNet
{
    CLImageLayerData::CLImageLayerData(size_t train_num,
            size_t width, size_t height, size_t map_num,
            CLImageLayerData::Channel ch)
        : CLLayerData(train_num, map_num*width*height),
            m_width(width), m_height(height), m_map(map_num), m_ch(ch)
    {
        cl::ImageFormat imgfmt;
        switch (m_ch)
        {
        case Channel::INTENSITY:
            imgfmt = {CL_INTENSITY, CL_FLOAT};
            break;
        }

        auto context = CLContext::getInstance().getContext();
        for (size_t i = 0; i < LayerData::DATA_COUNT * train_num; i++)
        {
            m_images.emplace_back(context, CL_MEM_READ_WRITE,
                    imgfmt, m_width, m_height, m_map);
        }

        m_origin[0] = m_origin[1] = m_origin[2] = 0;
        m_region[0] = m_width;
        m_region[1] = m_height;
        m_region[2] = m_map;
    }

    CLImageLayerData::~CLImageLayerData()
    {
    }

    void CLImageLayerData::loadToCL(DataIndex idx)
    {
        auto queue = CLContext::getInstance().getCommandQueue();
        cl_int err;
        std::vector<cl::Event> writeEvents;

        for (size_t i = 0; i < getTrainNum(); i++)
        {
            cl::Event ev;
            err = queue.enqueueWriteImage(
                    m_images[static_cast<int>(idx) * getTrainNum() + i],
                    CL_FALSE,
                    m_origin, m_region, 0, 0,
                    get(idx) + (getDataNum() * i),
                    nullptr, &ev);
            printError(err, "Error at CommandQueue::enqueueWriteImage in "
                    "CLImageLayerData::loadToCL");
            writeEvents.push_back(std::move(ev));
        }
        
        err = queue.enqueueMarkerWithWaitList(&writeEvents);
        printError(err, "Error at CommandQueue::enqueueMarkerWithWaitList in "
                "CLImageLayerData::loadToCL");
    }

    void CLImageLayerData::getFromCL(DataIndex idx)
    {
        auto queue = CLContext::getInstance().getCommandQueue();
        cl_int err;
        std::vector<cl::Event> readEvents;

        for (size_t i = 0; i < getTrainNum(); i++)
        {
            cl::Event ev;
            err = queue.enqueueReadImage(
                    m_images[static_cast<int>(idx) * getTrainNum() + i],
                    CL_FALSE,
                    m_origin, m_region, 0, 0,
                    get(idx) + (getDataNum() * i),
                    nullptr, &ev);
            printError(err, "Error at CommandQueue::enqueueReadBuffer in "
                    "CLImageLayerData::getFromCL");
            readEvents.push_back(std::move(ev));
        }
        
        err = queue.enqueueMarkerWithWaitList(&readEvents);
        printError(err, "Error at CommandQueue::enqueueMarkerWithWaitList in "
                "CLImageLayerData::getFromCL");
    }

    cl::Memory CLImageLayerData::getCLMemory(LayerData::DataIndex data_idx,
            size_t train_idx) const
    {
        return m_images[static_cast<int>(data_idx) * getTrainNum()
                + train_idx];
    }
}
