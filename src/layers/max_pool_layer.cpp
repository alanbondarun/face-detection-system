#include "layers/max_pool_layer.hpp"
#include "layers/cl_layer_data.hpp"
#include "layers/cl_image_layer_data.hpp"
#include "calc/calc-cpu.hpp"
#include "utils/cl_exception.hpp"
#include "utils/make_unique.hpp"
#include "cl_context.hpp"

namespace NeuralNet
{
    MaxPoolLayer::MaxPoolLayer(const Dimension& dim)
        : m_dim(dim),
        m_output_width((dim.image_width - dim.pool_width) /
                (dim.pool_width - (dim.stride - 1)) + 1),
        m_output_height((dim.image_height - dim.pool_height) /
                (dim.pool_height - (dim.stride - 1)) + 1)
    {
        if (m_dim.uses_gpu)
        {
            // create kernel
            m_fwd_kernel = cl::Kernel(CLContext::getInstance().getProgram(), "max_pool_forward");

            int pool_width = m_dim.pool_width;
            int pool_height = m_dim.pool_height;
            int stride = m_dim.stride;
            m_fwd_kernel.setArg(4, sizeof(int), &pool_width);
            m_fwd_kernel.setArg(5, sizeof(int), &pool_height);
            m_fwd_kernel.setArg(6, sizeof(int), &stride);
        }
    }

    void MaxPoolLayer::forward_cpu(const LayerData& prev, LayerData& current)
    {
        auto train_num = current.getTrainNum();
        auto prev_a = prev.get(LayerData::DataIndex::ACTIVATION);
        auto prev_z = prev.get(LayerData::DataIndex::INTER_VALUE);
        auto cur_a = current.get(LayerData::DataIndex::ACTIVATION);
        auto cur_z = current.get(LayerData::DataIndex::INTER_VALUE);

        for (size_t i=0; i<train_num; i++)
        {
            for (size_t j=0; j<m_dim.map_num; j++)
            {
                size_t back_offset = (i * m_dim.map_num + j)
                        * (m_dim.image_width * m_dim.image_height);
                size_t front_offset = (i * m_dim.map_num + j)
                        * (m_output_width * m_output_height);
                downsample_max(prev_z + back_offset, cur_z + front_offset,
                        m_dim.image_width, m_dim.image_height,
                        m_dim.pool_width, m_dim.pool_height, m_dim.stride);
                downsample_max(prev_a + back_offset, cur_a + front_offset,
                        m_dim.image_width, m_dim.image_height,
                        m_dim.pool_width, m_dim.pool_height, m_dim.stride);
            }
        }
    }

    void MaxPoolLayer::forward_gpu(const CLLayerData& prev, CLLayerData& current)
    {
        auto train_num = current.getTrainNum();

        for (size_t i=0; i<train_num; i++)
        {
            auto m_buf_pa = prev.getCLMemory(
                    LayerData::DataIndex::ACTIVATION, i);
            auto m_buf_pz = prev.getCLMemory(
                    LayerData::DataIndex::INTER_VALUE, i);
            auto m_buf_ca = current.getCLMemory(
                    LayerData::DataIndex::ACTIVATION, i);
            auto m_buf_cz = current.getCLMemory(
                    LayerData::DataIndex::INTER_VALUE, i);
            m_fwd_kernel.setArg(0, m_buf_pz);
            m_fwd_kernel.setArg(1, m_buf_pa);
            m_fwd_kernel.setArg(2, m_buf_cz);
            m_fwd_kernel.setArg(3, m_buf_ca);

            auto queue = CLContext::getInstance().getCommandQueue();
            cl_int err = CL_SUCCESS;

            err = queue.enqueueNDRangeKernel(m_fwd_kernel, cl::NullRange,
                    cl::NDRange(m_output_width, m_output_height,
                        m_dim.map_num),
                    cl::NullRange);
            printError(err, "Error at CommandQueue::enqueNDRangeKernel");
        }
    }

    void MaxPoolLayer::backward_cpu(LayerData& prev, LayerData& current)
    {
        auto train_num = current.getTrainNum();
        auto prev_e = prev.get(LayerData::DataIndex::ERROR);
        auto prev_a = prev.get(LayerData::DataIndex::ACTIVATION);
        auto cur_e = current.get(LayerData::DataIndex::ERROR);

        for (size_t i=0; i<train_num; i++)
        {
            for (size_t j=0; j<m_dim.map_num; j++)
            {
                size_t back_offset = (i * m_dim.map_num + j)
                        * (m_dim.image_width * m_dim.image_height);
                size_t front_offset = (i * m_dim.map_num + j)
                        * (m_output_width * m_output_height);

                upsample_max(cur_e + front_offset, prev_a + back_offset,
                        prev_e + back_offset,
                        m_dim.image_width, m_dim.image_height,
                        m_dim.pool_width, m_dim.pool_height, m_dim.stride);
            }
        }
    }

    void MaxPoolLayer::backward_gpu(CLLayerData& prev, CLLayerData& current)
    {
        /* TODO: OpenCL intergration */
        // not implemented yet, just use cpu temporarily
        backward_cpu(prev, current);
    }

    std::unique_ptr<LayerData> MaxPoolLayer::createLayerData(size_t train_num)
    {
        if (m_dim.uses_gpu)
        {
            return std::make_unique<CLImageLayerData>(
                train_num,
                m_output_width, m_output_height, m_dim.map_num,
                CLImageLayerData::Channel::INTENSITY
            );
        }
        return std::make_unique<LayerData>(
            train_num,
            m_dim.map_num * m_output_width * m_output_height
        );
    }

    size_t MaxPoolLayer::getNeuronNum() const
    {
        return m_dim.map_num * m_output_width * m_output_height;
    }

    void MaxPoolLayer::importLayer(const Json::Value& coeffs)
    {
    }

    Json::Value MaxPoolLayer::exportLayer()
    {
        return Json::Value();
    }
}
