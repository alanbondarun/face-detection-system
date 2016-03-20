#ifndef __CL_LAYER_DATA_HPP
#define __CL_LAYER_DATA_HPP

#include "layers/layer_data.hpp"
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

namespace NeuralNet
{
    class CLLayerData: public LayerData
    {
    public:
        CLLayerData(size_t train_num, size_t data_num);
        virtual ~CLLayerData();

        void loadToCLBuffer(DataIndex idx);
        void getFromCLBuffer(DataIndex idx);

        cl::Buffer getCLBuffer(LayerData::DataIndex idx) const;

    private:
        std::vector<cl::Buffer> m_buffers;
    };
}

#endif // __CL_LAYER_DATA_HPP
