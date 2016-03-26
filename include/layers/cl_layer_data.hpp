#ifndef __CL_LAYER_DATA_HPP
#define __CL_LAYER_DATA_HPP

#include "layers/layer_data.hpp"
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/cl.hpp"

namespace NeuralNet
{
    class CLLayerData: public LayerData
    {
    public:
        CLLayerData(size_t train_num, size_t data_num)
            : LayerData(train_num, data_num) {}
        virtual ~CLLayerData() {}

        virtual void loadToCL(DataIndex idx) = 0;
        virtual void getFromCL(DataIndex idx) = 0;

        virtual cl::Memory getCLMemory(LayerData::DataIndex data_idx)
            const = 0;
    };
}

#endif // __CL_LAYER_DATA_HPP
