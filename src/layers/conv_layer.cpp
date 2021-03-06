#include "layers/conv_layer.hpp"
#include "layers/cl_layer_data.hpp"
#include "layers/cl_image_layer_data.hpp"
#include "calc/calc-cpu.hpp"
#include "calc/util-functions.hpp"
#include "utils/make_unique.hpp"
#include "utils/cl_exception.hpp"
#include "cl_context.hpp"
#include <random>
#include <cstring>
#include <cmath>
#include <array>
#include <iostream>

namespace NeuralNet
{
    ConvLayer::ConvLayer(const LayerSetting& setting, ActivationFunc func)
        : m_set(setting), m_learn_rate(setting.learn_rate)
    {
        switch (func)
        {
        case ActivationFunc::SIGMOID:
            f_activation = ActivationFuncs::f_sigmoid;
            f_activation_prime = ActivationFuncs::f_sigmoid_prime;
            break;
        case ActivationFunc::RELU:
            f_activation = ActivationFuncs::f_relu;
            f_activation_prime = ActivationFuncs::f_relu_prime;
            break;
        }

        if (m_set.enable_zero_pad)
        {
            f_convolution = convolution_mat_same_zeros;
            f_convol_back = convolution_mat_same_zeros;
            m_output_width = m_set.image_width;
            m_output_height = m_set.image_height;
        }
        else
        {
            f_convolution = convolution_mat_no_zeros;
            f_convol_back = convolution_mat_wide_zeros;
            m_output_width = m_set.image_width - (m_set.recep_size - 1);
            m_output_height = m_set.image_height - (m_set.recep_size - 1);
        }

        const size_t num_weights = m_set.current_map_num * m_set.prev_map_num
                * m_set.recep_size * m_set.recep_size;
        m_weight = new float[num_weights];

        const size_t num_biases = m_set.current_map_num * m_output_width
                * m_output_height;
        m_bias = new float[num_biases];

        // weight and bias initialization
        std::random_device rd;
        std::mt19937 rgen(rd());

        std::normal_distribution<float> dist_w(0.0, std::sqrt(2.0 / (m_set.image_width * m_set.image_height)));
        std::normal_distribution<float> dist_b(0.0, std::sqrt(2.0 / (m_set.image_width * m_set.image_height)));

        for (size_t i = 0; i < num_weights; i++)
            m_weight[i] = dist_w(rgen);
        for (size_t i = 0; i < num_biases; i++)
            m_bias[i] = dist_b(rgen);

        if (m_set.uses_gpu)
        {
            // initialize buffers
            cl::Context context = CLContext::getInstance().getContext();
            cl::ImageFormat value_fmt{CL_INTENSITY, CL_FLOAT};

            m_imgbuf_w = cl::Image3D(context, CL_MEM_READ_ONLY, value_fmt,
                    m_set.recep_size * m_set.recep_size,
                    m_set.prev_map_num,
                    m_set.current_map_num);
            m_imgbuf_b = cl::Image3D(context, CL_MEM_READ_ONLY, value_fmt,
                    m_output_width,
                    m_output_height,
                    m_set.current_map_num);

            // create kernel (TODO: non-ReLU activation function?)
            if (m_set.enable_zero_pad)
            {
                m_fwd_kernel = cl::Kernel(CLContext::getInstance().getProgram(),
                    "conv_forward_relu_zeropad");
            }
            else
            {
                // TODO: non-zeropad case?
            }

            m_fwd_kernel.setArg(3, m_imgbuf_w);
            m_fwd_kernel.setArg(4, m_imgbuf_b);

            int i_in_width = m_set.image_width;
            int i_out_width = m_output_width;
            m_fwd_kernel.setArg(5, sizeof(int), &i_in_width);
            m_fwd_kernel.setArg(6, sizeof(int), &i_out_width);

            refreshCLLayerInfo();
        }
    }

    ConvLayer::~ConvLayer()
    {
        delete [] m_bias;
        delete [] m_weight;
    }

    void ConvLayer::forward_cpu(const LayerData& prev, LayerData& current)
    {
        auto train_num = current.getTrainNum();
        auto prev_a = prev.get(LayerData::DataIndex::ACTIVATION);
        auto cur_a = current.get(LayerData::DataIndex::ACTIVATION);
        auto cur_z = current.get(LayerData::DataIndex::INTER_VALUE);

        memset(cur_z, 0, sizeof(float) * train_num
                * m_set.current_map_num * m_output_width * m_output_height);
        float *temp_z = new float[m_output_width * m_output_height];

        for (size_t i=0; i<train_num; i++)
        {
            size_t w_offset = 0;
            size_t prev_offset = 0;
            size_t cur_offset = i * m_set.current_map_num * m_output_width
                * m_output_height;
            size_t bias_offset = 0;

            for (size_t ncur = 0; ncur < m_set.current_map_num; ncur++)
            {
                prev_offset = i * m_set.prev_map_num * m_set.image_width * m_set.image_height;
                for (size_t nprev = 0; nprev < m_set.prev_map_num; nprev++)
                {
                    f_convolution(
                        prev_a + prev_offset, m_weight + w_offset, temp_z,
                        m_set.image_width, m_set.image_height, m_set.recep_size, m_set.recep_size
                    );
                    add_vec(cur_z + cur_offset, temp_z, cur_z + cur_offset, m_output_width * m_output_height);

                    w_offset += (m_set.recep_size * m_set.recep_size);
                    prev_offset += (m_set.image_width * m_set.image_height);
                }

                add_vec(cur_z + cur_offset, m_bias + bias_offset, cur_z + cur_offset,
                        m_output_width * m_output_height);
                apply_vec(cur_z + cur_offset, cur_a + cur_offset,
                        m_output_width * m_output_height, f_activation);

                cur_offset += (m_output_width * m_output_height);
                bias_offset += (m_output_width * m_output_height);
            }
        }

        delete [] temp_z;
    }

    void ConvLayer::forward_gpu(const CLLayerData& prev, CLLayerData& current)
    {
        auto queue = CLContext::getInstance().getCommandQueue();

        auto m_buf_pa = prev.getCLMemory(
                LayerData::DataIndex::ACTIVATION);
        auto m_buf_cz = current.getCLMemory(
                LayerData::DataIndex::INTER_VALUE);
        auto m_buf_ca = current.getCLMemory(
                LayerData::DataIndex::ACTIVATION);
        m_fwd_kernel.setArg(0, m_buf_pa);
        m_fwd_kernel.setArg(1, m_buf_cz);
        m_fwd_kernel.setArg(2, m_buf_ca);

        cl_int err = CL_SUCCESS;
        err = queue.enqueueNDRangeKernel(m_fwd_kernel, cl::NullRange,
                cl::NDRange(m_output_width * m_output_height,
                    m_set.current_map_num,
                    current.getTrainNum()),
                cl::NullRange);
        printError(err, "Error at CommandQueue::enqueNDRangeKernel");
    }

    void ConvLayer::backward_cpu(LayerData& prev, LayerData& current)
    {
        const auto train_num = current.getTrainNum();
        const auto learn_rate = m_learn_rate;
        const int i_recep_size = m_set.recep_size;
        auto prev_a = prev.get(LayerData::DataIndex::ACTIVATION);
        auto prev_z = prev.get(LayerData::DataIndex::INTER_VALUE);
        auto prev_e = prev.get(LayerData::DataIndex::ERROR);
        auto cur_e = current.get(LayerData::DataIndex::ERROR);

        memset(prev_e, 0, sizeof(float) * train_num
                * m_set.prev_map_num * m_set.image_width * m_set.image_height);

        /* calculate error value for previous layer */
        float *sprime_z = new float[m_set.image_width * m_set.image_height];
        float *temp_w = new float[m_set.recep_size * m_set.recep_size];
        float *temp_pe = new float[m_set.image_width * m_set.image_height];
        for (size_t i=0; i<train_num; i++)
        {
            size_t w_offset = 0;
            size_t cur_offset = 0;
            size_t prev_offset = i * m_set.prev_map_num * m_set.image_width * m_set.image_height;
            for (size_t nprev = 0; nprev < m_set.prev_map_num; nprev++)
            {
                cur_offset = i * m_set.current_map_num * m_output_width * m_output_height;
                w_offset = nprev * m_set.recep_size * m_set.recep_size;

                for (size_t ncur = 0; ncur < m_set.current_map_num; ncur++)
                {
                    flip_mat(m_weight + w_offset, temp_w, m_set.recep_size, m_set.recep_size);
                    f_convol_back(
                        cur_e + cur_offset, temp_w, temp_pe,
                        m_output_width, m_output_height, m_set.recep_size, m_set.recep_size
                    );
                    add_vec(prev_e + prev_offset, temp_pe, prev_e + prev_offset,
                            m_set.image_width * m_set.image_height);

                    w_offset += (m_set.prev_map_num * m_set.recep_size * m_set.recep_size);
                    cur_offset += (m_output_width * m_output_height);
                }

                apply_vec(prev_z + prev_offset, sprime_z, m_set.image_width * m_set.image_height,
                        f_activation_prime);
                pmul_vec(prev_e + prev_offset, sprime_z, prev_e + prev_offset,
                        m_set.image_width * m_set.image_height);

                prev_offset += (m_set.image_width * m_set.image_height);
            }
        }

        // add weight decay term
        const_mul_vec(m_weight, 1.0 - (m_set.learn_rate * m_set.weight_decay),
                m_set.current_map_num * m_set.prev_map_num *
                m_set.recep_size * m_set.recep_size);

        /* calculate delta_w and update current weight */
        float *delta_w = new float[m_set.recep_size * m_set.recep_size];
        size_t dw_offset = 0;
        for (size_t ncur = 0; ncur < m_set.current_map_num; ncur++)
        {
            for (size_t nprev = 0; nprev < m_set.prev_map_num; nprev++)
            {
                size_t cur_offset = ncur * m_output_width * m_output_height;
                size_t prev_offset = (nprev * m_set.image_width * m_set.image_height);

                memset(delta_w, 0, sizeof(float) * m_set.recep_size * m_set.recep_size);

                for (size_t i = 0; i < train_num; i++)
                {
                    if (m_set.enable_zero_pad)
                    {
                        convolution_mat(prev_a + prev_offset, cur_e + cur_offset, temp_w,
                            m_set.image_width, m_set.image_height, m_output_width, m_output_height,
                            MatrixRange(-(i_recep_size/2), -(i_recep_size/2), i_recep_size, i_recep_size));
                    }
                    else
                    {
                        convolution_mat_no_zeros(prev_a + prev_offset, cur_e + cur_offset, temp_w,
                            m_set.image_width, m_set.image_height, m_output_width, m_output_height);
                    }
                    add_vec(delta_w, temp_w, delta_w, m_set.recep_size * m_set.recep_size);

                    prev_offset += (m_set.prev_map_num * m_set.image_width * m_set.image_height);
                    cur_offset += (m_set.current_map_num * m_output_width * m_output_height);
                }

                apply_vec(delta_w, delta_w, m_set.recep_size * m_set.recep_size,
                    [train_num, learn_rate](float in) -> float {
                        return -in*learn_rate/train_num;
                });
                add_vec(m_weight + dw_offset, delta_w, m_weight + dw_offset,
                        m_set.recep_size * m_set.recep_size);

                dw_offset += (m_set.recep_size * m_set.recep_size);
            }
        }

        // calculate delta_b and update current bias
        float *delta_b = new float[m_set.current_map_num * m_output_width *
                m_output_height];
        sum_vec(cur_e, delta_b, m_set.current_map_num * m_output_width * m_output_height,
                train_num);
        apply_vec(delta_b, delta_b,
                m_set.current_map_num * m_output_width * m_output_height,
                [train_num, learn_rate](float in) -> float {
                    return -in*learn_rate/train_num;
                });
        add_vec(m_bias, delta_b, m_bias,
                m_set.current_map_num * m_output_width * m_output_height);

        delete [] delta_b;
        delete [] delta_w;
        delete [] temp_pe;
        delete [] temp_w;
        delete [] sprime_z;
    }

    void ConvLayer::backward_gpu(CLLayerData& prev, CLLayerData& current)
    {
        /* TODO: OpenCL intergration */
        // not implemented yet, just use cpu temporarily
        backward_cpu(prev, current);

        refreshCLLayerInfo();
    }

    void ConvLayer::refreshCLLayerInfo()
    {
        if (m_set.uses_gpu)
        {
            auto queue = CLContext::getInstance().getCommandQueue();
            cl_int err = CL_SUCCESS;

            cl::size_t<3> in_offset, in_region_w, in_region_b;
            in_offset[0] = in_offset[1] = in_offset[2] = 0;
            in_region_w[0] = m_set.recep_size * m_set.recep_size;
            in_region_w[1] = m_set.prev_map_num;
            in_region_w[2] = m_set.current_map_num;
            in_region_b[0] = m_output_width;
            in_region_b[1] = m_output_height;
            in_region_b[2] = m_set.current_map_num;

            err = queue.enqueueWriteImage(m_imgbuf_w, CL_TRUE, in_offset,
                    in_region_w, 0, 0, m_weight);
            printError(err, "Error at CommandQueue::enqueueWriteBuffer for m_buf_w");
            err = queue.enqueueWriteImage(m_imgbuf_b, CL_TRUE, in_offset,
                    in_region_b, 0, 0, m_bias);
            printError(err, "Error at CommandQueue::enqueueWriteBuffer for m_buf_b");
        }
    }

    std::unique_ptr<LayerData> ConvLayer::createLayerData(size_t train_num)
    {
        if (m_set.uses_gpu)
        {
            return std::make_unique<CLImageLayerData>(
                    train_num,
                    m_output_width, m_output_height, m_set.current_map_num,
                    CLImageLayerData::Channel::INTENSITY
            );
        }
        return std::make_unique<LayerData>(
            train_num,
            m_set.current_map_num * m_output_width * m_output_height
        );
    }
    
    size_t ConvLayer::getNeuronNum() const
    {
        return m_set.current_map_num * m_output_width * m_output_height;
    }

    void ConvLayer::importLayer(const Json::Value& coeffs)
    {
        auto coeff_dim = coeffs["dimension"];
        size_t cur_maps = coeff_dim["map_num"].asUInt();
        if (cur_maps != m_set.current_map_num)
            throw Json::LogicError("invalid map_num");
        size_t recep_size = coeff_dim["recep_size"].asUInt();
        if (recep_size != m_set.recep_size)
            throw Json::LogicError("invalid recep_size");

        size_t weight_offset = 0;
        auto weight_lists = coeffs["weight"];
        if (weight_lists.size() != m_set.current_map_num * m_set.prev_map_num)
            throw Json::LogicError("invalid number of feature maps");
        for (size_t i = 0; i < m_set.current_map_num * m_set.prev_map_num; i++)
        {
            auto weights = weight_lists[static_cast<int>(i)];
            if (weights.size() != m_set.recep_size * m_set.recep_size)
                throw Json::LogicError("invalid feature map");

            for (size_t j = 0; j < m_set.recep_size * m_set.recep_size; j++)
            {
                *(m_weight + weight_offset) = weights[static_cast<int>(j)].asDouble();
                weight_offset++;
            }
        }

        size_t bias_offset = 0;
        auto bias_lists = coeffs["bias"];
        if (bias_lists.size() != m_set.current_map_num)
            throw Json::LogicError("invalid number of feature maps");
        for (size_t i = 0; i < m_set.current_map_num; i++)
        {
            auto biases = bias_lists[static_cast<int>(i)];
            if (biases.size() != m_output_width * m_output_height)
                throw Json::LogicError("invalid biases for a feature map");

            for (size_t j = 0; j < m_output_width * m_output_height; j++)
            {
                *(m_bias + bias_offset) = biases[static_cast<int>(j)].asDouble();
                bias_offset++;
            }
        }

        refreshCLLayerInfo();
    }

    Json::Value ConvLayer::exportLayer()
    {
        Json::Value coeff_value(Json::objectValue);

        Json::Value coeff_dim(Json::objectValue);
        coeff_dim["map_num"] = Json::Value(static_cast<Json::UInt>(m_set.current_map_num));
        coeff_dim["recep_size"] = Json::Value(static_cast<Json::UInt>(m_set.recep_size));
        coeff_value["dimension"] = coeff_dim;

        size_t weight_offset = 0;
        Json::Value weight_lists(Json::arrayValue);
        for (size_t i = 0; i < m_set.current_map_num * m_set.prev_map_num; i++)
        {
            Json::Value weights(Json::arrayValue);
            for (size_t j = 0; j < m_set.recep_size * m_set.recep_size; j++)
            {
                weights.append(Json::Value(m_weight[weight_offset]));
                weight_offset++;
            }

            weight_lists.append(weights);
        }
        coeff_value["weight"] = weight_lists;

        size_t bias_offset = 0;
        Json::Value bias_lists(Json::arrayValue);
        for (size_t i = 0; i < m_set.current_map_num; i++)
        {
            Json::Value biases(Json::arrayValue);
            for (size_t j = 0; j < m_output_width * m_output_height; j++)
            {
                biases.append(Json::Value(m_bias[bias_offset]));
                bias_offset++;
            }

            bias_lists.append(biases);
        }
        coeff_value["bias"] = bias_lists;

        return coeff_value;
    }
}
