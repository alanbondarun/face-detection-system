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
        CLLayerData(size_t train_num, size_t data_num);
        virtual ~CLLayerData();

        void loadToCLBuffer(DataIndex idx);
        void getFromCLBuffer(DataIndex idx);

        cl::Buffer getCLBuffer(LayerData::DataIndex idx) const;

    protected:
        virtual void updateDataSize(size_t new_train_num);

    private:
        std::vector<cl::Buffer> m_buffers;
    };
}

#endif // __CL_LAYER_DATA_HPP
