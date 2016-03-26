#include "layers/sigmoid_layer.hpp"
#include "calc/calc-cpu.hpp"
#include "calc/util-functions.hpp"
#include "utils/make_unique.hpp"
#include "utils/cl_exception.hpp"
#include "cl_context.hpp"
#include <cstring>
#include <random>
#include <cmath>

namespace NeuralNet
{
    SigmoidLayer::SigmoidLayer(const Setting& set)
        : m_prev_d(set.prev_neurons), m_current_d(set.current_neurons),
        m_learn_rate(set.learn_rate),
        m_uses_dropout(set.dropout_enable), m_dropout_enabled(false),
        m_dropout_rate(set.dropout_rate),
        m_uses_gpu(set.uses_gpu),
        m_weight_decay(set.weight_decay)
    {
        m_weight = new float[m_current_d * m_prev_d];
        m_bias = new float[m_current_d];

        m_dropout_coeff = new float[m_current_d];

        /* weight and bias initializaion */
        std::random_device rd;
        std::mt19937 rgen(rd());
        std::normal_distribution<float> dist_w(0.0, std::sqrt(2.0 / (m_prev_d)));
        std::normal_distribution<float> dist_b(0.0, std::sqrt(2.0 / (m_prev_d)));

        for (size_t i = 0; i < m_current_d * m_prev_d; i++)
            m_weight[i] = dist_w(rgen);
        for (size_t i = 0; i < m_current_d; i++)
            m_bias[i] = dist_b(rgen);

        // dropout coefficient init
        m_dropout_enabled = false;
        if (m_uses_dropout)
        {
            set_vec(m_dropout_coeff, m_dropout_rate, m_current_d);
        }
        else
        {
            set_vec(m_dropout_coeff, 1.0, m_current_d);
        }

        if (m_uses_gpu)
        {
            // initialize buffers
            cl::Context context = CLContext::getInstance().getContext();
            cl::ImageFormat weight_fmt{CL_INTENSITY, CL_FLOAT};
            cl_int err;

            m_imgbuf_w = cl::Image2D(context, CL_MEM_READ_ONLY, weight_fmt,
                    m_prev_d, m_current_d, 0, nullptr, &err);
            printError(err, "m_imgbuf_w in SigmoidLayer");

            m_buf_b = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * m_current_d);
            m_buf_do = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * m_current_d);

            // create kernel
            m_fwd_kernel = cl::Kernel(CLContext::getInstance().getProgram(), "sigmoid_forward");
            m_fwd_img_kernel = cl::Kernel(CLContext::getInstance().getProgram(),
                    "sigmoid_forward_img");

            m_fwd_kernel.setArg(1, m_imgbuf_w);
            m_fwd_kernel.setArg(2, m_buf_b);
            m_fwd_kernel.setArg(5, m_buf_do);
            m_fwd_img_kernel.setArg(1, m_imgbuf_w);
            m_fwd_img_kernel.setArg(2, m_buf_b);
            m_fwd_img_kernel.setArg(5, m_buf_do);

            int i_prev_d = static_cast<int>(m_prev_d);
            int i_cur_d = static_cast<int>(m_current_d);
            m_fwd_kernel.setArg(6, sizeof(int), &i_prev_d);
            m_fwd_kernel.setArg(7, sizeof(int), &i_cur_d);
            m_fwd_img_kernel.setArg(6, sizeof(int), &i_prev_d);
            m_fwd_img_kernel.setArg(7, sizeof(int), &i_cur_d);

            refreshCLLayerInfo();
            updateDOBuffer();
        }
    }

    SigmoidLayer::~SigmoidLayer()
    {
        delete [] m_dropout_coeff;

        delete [] m_bias;
        delete [] m_weight;
    }

    void SigmoidLayer::forward_cpu(const LayerData& prev, LayerData& current)
    {
        /* TODO: data correctness check? */

        auto m_train_num = current.getTrainNum();
        auto prev_a = prev.get(LayerData::DataIndex::ACTIVATION);
        auto cur_z = current.get(LayerData::DataIndex::INTER_VALUE);
        auto cur_a = current.get(LayerData::DataIndex::ACTIVATION);

        for (size_t i=0; i<m_train_num; i++)
        {
            mul_mat_vec(m_weight, prev_a + (i*m_prev_d), cur_z + (i*m_current_d),
                    m_current_d, m_prev_d);
            add_vec(cur_z + (i*m_current_d), m_bias,
                    cur_z + (i*m_current_d), m_current_d);
        }
        apply_vec(cur_z, cur_a, m_current_d * m_train_num, ActivationFuncs::f_sigmoid);

        refreshDropout();

        if (m_uses_dropout)
        {
            for (size_t m = 0; m < m_train_num; m++)
            {
                pmul_vec(cur_a + (m_current_d*m), m_dropout_coeff,
                        cur_a + (m_current_d*m), m_current_d);
            }
        }
    }

    void SigmoidLayer::forward_gpu(const CLLayerData& prev, CLLayerData& current)
    {
        auto* prev_ptr = &prev;
        auto* current_ptr = &current;
        auto* cbptr = dynamic_cast<CLBufferLayerData*>(current_ptr);
        
        if (cbptr)
        {
            auto* pbptr = dynamic_cast<const CLBufferLayerData*>(prev_ptr);
            auto* piptr = dynamic_cast<const CLImageLayerData*>(prev_ptr);
            if (pbptr)
            {
                forward_gpu(*pbptr, *cbptr);
            }
            else if (piptr)
            {
                forward_gpu(*piptr, *cbptr);
            }
        }
    }

    void SigmoidLayer::forward_gpu(const CLImageLayerData& prev,
            CLBufferLayerData& current)
    {
        refreshDropout();

        cl::CommandQueue queue = CLContext::getInstance().getCommandQueue();

        auto m_buf_pa = prev.getCLMemory(
                LayerData::DataIndex::ACTIVATION);
        auto m_buf_cz = current.getCLMemory(
                LayerData::DataIndex::INTER_VALUE);
        auto m_buf_ca = current.getCLMemory(
                LayerData::DataIndex::ACTIVATION);

        m_fwd_img_kernel.setArg(0, m_buf_pa);
        m_fwd_img_kernel.setArg(3, m_buf_cz);
        m_fwd_img_kernel.setArg(4, m_buf_ca);

        cl_int err = CL_SUCCESS;
        err = queue.enqueueNDRangeKernel(m_fwd_img_kernel, cl::NullRange,
                cl::NDRange(m_current_d, current.getTrainNum()),
                cl::NullRange);
        printError(err, "Error at CommandQueue::enqueNDRangeKernel");
    }

    void SigmoidLayer::forward_gpu(const CLBufferLayerData& prev,
            CLBufferLayerData& current)
    {
        refreshDropout();

        cl::CommandQueue queue = CLContext::getInstance().getCommandQueue();

        auto m_buf_pa = prev.getCLMemory(
                LayerData::DataIndex::ACTIVATION);
        auto m_buf_cz = current.getCLMemory(
                LayerData::DataIndex::INTER_VALUE);
        auto m_buf_ca = current.getCLMemory(
                LayerData::DataIndex::ACTIVATION);

        m_fwd_kernel.setArg(0, m_buf_pa);
        m_fwd_kernel.setArg(3, m_buf_cz);
        m_fwd_kernel.setArg(4, m_buf_ca);

        cl_int err = CL_SUCCESS;
        err = queue.enqueueNDRangeKernel(m_fwd_kernel, cl::NullRange,
                cl::NDRange(m_current_d, current.getTrainNum()),
                cl::NullRange);
        printError(err, "Error at CommandQueue::enqueNDRangeKernel");
    }

    void SigmoidLayer::backward_cpu(LayerData& prev, LayerData& current)
    {
        /* TODO: data correctness check? */
        const auto m_train_num = current.getTrainNum();
        const auto train_num = m_train_num;
        const auto learn_rate = m_learn_rate;
        auto prev_e = prev.get(LayerData::DataIndex::ERROR);
        auto prev_z = prev.get(LayerData::DataIndex::INTER_VALUE);
        auto prev_a = prev.get(LayerData::DataIndex::ACTIVATION);
        auto cur_e = current.get(LayerData::DataIndex::ERROR);

        if (m_uses_dropout)
        {
            // dropout is applied only to the error value of the dropout-enabled layer
            // in the backpropagation phase
            if (m_dropout_enabled)
            {
                for (size_t m = 0; m < train_num; m++)
                {
                    pmul_vec(cur_e + (m_current_d*m), m_dropout_coeff,
                            cur_e + (m_current_d*m), m_current_d);
                }
            }
            else
            {
                const_mul_vec(cur_e, m_dropout_rate, m_train_num * m_current_d);
            }
        }

        /* calculate error value for previous layer */
        float *sprime_z = new float[m_prev_d * m_train_num];
        apply_vec(prev_z, sprime_z, m_prev_d * m_train_num, ActivationFuncs::f_sigmoid_prime);

        float *temp_w = new float[m_prev_d * m_current_d];
        transpose_mat(m_weight, temp_w, m_current_d, m_prev_d);

        for (size_t i=0; i<m_train_num; i++)
        {
            mul_mat_vec(temp_w, cur_e + (i*m_current_d), prev_e + (i*m_prev_d), m_prev_d, m_current_d);
        }
        pmul_vec(prev_e, sprime_z, prev_e, m_prev_d * m_train_num);

        /* calculate delta_b and update current bias */
        float *delta_b = new float[m_current_d];

        sum_vec(cur_e, delta_b, m_current_d, m_train_num);
        apply_vec(delta_b, delta_b, m_current_d, [train_num, learn_rate](float in) -> float {
            return -in*learn_rate/train_num;
        });
        add_vec(m_bias, delta_b, m_bias, m_current_d);

        // add decay term
        const_mul_vec(m_weight, 1.0 - m_learn_rate * m_weight_decay, m_prev_d * m_current_d);

        /* calculate delta_w and update current weight */
        float *delta_w = new float[m_prev_d * m_current_d];
        memset(delta_w, 0, sizeof(float) * m_prev_d * m_current_d);

        for (size_t i=0; i<m_train_num; i++)
        {
            vec_outer_prod(cur_e, prev_a, temp_w, m_current_d, m_prev_d);
            add_vec(delta_w, temp_w, delta_w, m_prev_d * m_current_d);
        }
        apply_vec(delta_w, delta_w, m_prev_d * m_current_d, [train_num, learn_rate](float in) -> float {
            return -in*learn_rate/train_num;
        });
        add_vec(m_weight, delta_w, m_weight, m_prev_d * m_current_d);


        delete [] delta_w;
        delete [] delta_b;
        delete [] temp_w;
        delete [] sprime_z;
    }

    void SigmoidLayer::backward_gpu(CLLayerData& prev, CLLayerData& current)
    {
        /* TODO */
        // not implemented yet, just use cpu temporarily
        backward_cpu(prev, current);
        refreshCLLayerInfo();
    }

    void SigmoidLayer::setDropout(bool enable)
    {
        // change dropout coefficients according to the settings
        if (m_uses_dropout)
        {
            if (enable != m_dropout_enabled)
            {
                m_dropout_enabled = enable;
                if (enable)
                {
                    refreshDropout();
                }
                else
                {
                    set_vec(m_dropout_coeff, m_dropout_rate, m_current_d);
                    updateDOBuffer();
                }
            }
        }
    }

    void SigmoidLayer::refreshDropout()
    {
        if (m_uses_dropout && m_dropout_enabled)
        {
            std::random_device rd;
            std::mt19937 rgen(rd());
            std::uniform_real_distribution<> dis(0, 1);
            const float dropout_rate = m_dropout_rate;

            // if this layer uses dropout, select neurons which does not participate in training
            // by setting their activation value to zero.
            apply_vec(m_dropout_coeff, m_dropout_coeff, m_current_d,
                    [dropout_rate, &rgen, &dis](float in) -> float {
                        return (dis(rgen) <= dropout_rate) ? 1:0;
                    });

            updateDOBuffer();
        }
    }

    void SigmoidLayer::updateDOBuffer()
    {
        if (m_uses_gpu)
        {
            cl::CommandQueue queue = CLContext::getInstance().getCommandQueue();
            cl_int err = CL_SUCCESS;
            err = queue.enqueueWriteBuffer(m_buf_do, CL_TRUE, 0, sizeof(float) * m_current_d,
                    m_dropout_coeff);
            printError(err, "Error at CommandQueue::enqueWriteBuffer for m_buf_do");
        }
    }

    void SigmoidLayer::refreshCLLayerInfo()
    {
        if (m_uses_gpu)
        {
            cl::CommandQueue queue = CLContext::getInstance().getCommandQueue();
            cl_int err = CL_SUCCESS;

            cl::size_t<3> in_offset, in_region;
            in_region[0] = m_prev_d;
            in_region[1] = m_current_d;
            in_region[2] = 1;

            err = queue.enqueueWriteImage(m_imgbuf_w, CL_TRUE, in_offset,
                    in_region, 0, 0, m_weight);
            printError(err, "Error at CommandQueue::enqueWriteBuffer for m_imgbuf_w");
            
            err = queue.enqueueWriteBuffer(m_buf_b, CL_TRUE, 0, sizeof(float) * m_current_d,
                    m_bias);
            printError(err, "Error at CommandQueue::enqueWriteBuffer for m_buf_b");
        }
    }

    std::unique_ptr<LayerData> SigmoidLayer::createLayerData(size_t train_num)
    {
        if (m_uses_gpu)
        {
            return std::make_unique<CLBufferLayerData>(
                    train_num,
                    m_current_d
            );
        }
        return std::make_unique<LayerData>(
            train_num,
            m_current_d
        );
    }

    size_t SigmoidLayer::getNeuronNum() const { return m_current_d; }

    void SigmoidLayer::importLayer(const Json::Value& coeffs)
    {
        size_t neurons = coeffs["neurons"].asUInt();
        if (neurons != m_current_d)
            throw Json::LogicError("invalid 'neurons' in layer_data");

        auto weight_lists = coeffs["weight"];
        if (weight_lists.size() != m_current_d)
            throw Json::LogicError("invalid number of weight values");
        for (int i = 0; i < static_cast<int>(m_current_d); i++)
        {
            auto weights = weight_lists[i];
            if (weights.size() != m_prev_d)
                throw Json::LogicError("invalid number of weight values");

            for (int j = 0; j < static_cast<int>(m_prev_d); j++)
                m_weight[i * m_prev_d + j] = weights[j].asDouble();
        }

        auto biases = coeffs["bias"];
        if (biases.size() != m_current_d)
            throw Json::LogicError("invalid number of bias values");
        for (int i = 0; i < static_cast<int>(m_current_d); i++)
            m_bias[i] = biases[i].asDouble();

        refreshCLLayerInfo();
    }

    Json::Value SigmoidLayer::exportLayer()
    {
        Json::Value coeff_value(Json::objectValue);
        coeff_value["neurons"] = Json::Value(static_cast<Json::UInt>(m_current_d));

        Json::Value weight_lists(Json::arrayValue);
        for (size_t i = 0; i < m_current_d; i++)
        {
            Json::Value weights(Json::arrayValue);
            for (size_t j = 0; j < m_prev_d; j++)
                weights.append(Json::Value(m_weight[i * m_prev_d + j]));

            weight_lists.append(weights);
        }
        coeff_value["weight"] = weight_lists;

        Json::Value biases(Json::arrayValue);
        for (size_t i = 0; i < m_current_d; i++)
            biases.append(Json::Value(m_bias[i]));
        coeff_value["bias"] = biases;

        return coeff_value;
    }
}
