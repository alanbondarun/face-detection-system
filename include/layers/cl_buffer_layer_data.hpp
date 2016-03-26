#ifndef __CL_BUFFER_LAYER_DATA_HPP
#define __CL_BUFFER_LAYER_DATA_HPP

#include "layers/cl_layer_data.hpp"
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/cl.hpp"

namespace NeuralNet
{
    class CLBufferLayerData: public CLLayerData
    {
    public:
        CLBufferLayerData(size_t train_num, size_t data_num);
        virtual ~CLBufferLayerData();

        virtual void loadToCL(DataIndex idx);
        virtual void getFromCL(DataIndex idx);

        virtual cl::Memory getCLMemory(LayerData::DataIndex data_idx,
                size_t train_idx) const;

    private:
        cl::Buffer mergeBuffers();

        std::vector<cl::Buffer> m_buffers;
    };
}

#endif // __CL_BUFFER_LAYER_DATA_HPP
