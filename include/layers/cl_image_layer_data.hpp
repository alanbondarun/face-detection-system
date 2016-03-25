#ifndef __CL_IMAGE_LAYER_DATA_HPP
#define __CL_IMAGE_LAYER_DATA_HPP

#include "layers/cl_layer_data.hpp"
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/cl.hpp"

namespace NeuralNet
{
    class CLImageLayerData: public CLLayerData
    {
    public:
        enum class Channel
        {
            INTENSITY,
        };

        CLImageLayerData(size_t train_num, size_t width, size_t height,
                size_t map_num, Channel ch);
        virtual ~CLImageLayerData();

        virtual void loadToCL(DataIndex idx);
        virtual void getFromCL(DataIndex idx);

        virtual cl::Memory getCLMemory(LayerData::DataIndex data_idx,
                size_t train_idx) const;

    private:
        const size_t m_width, m_height, m_map;
        const Channel m_ch;
        cl::size_t<3> m_origin, m_region;
        std::vector<cl::Image3D> m_images;
    };
}

#endif // __CL_IMAGE_LAYER_DATA_HPP
