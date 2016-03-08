#include "layers/sigmoid_layer.hpp"
#include "calc/calc-cpu.hpp"
#include "calc/util-functions.hpp"
#include "utils/make_unique.hpp"
#include <cstring>
#include <random>
#include <cmath>

namespace NeuralNet
{
    SigmoidLayer::SigmoidLayer(const Setting& set)
        : m_prev_d(set.prev_neurons), m_current_d(set.current_neurons),
        m_learn_rate(set.learn_rate),
        m_uses_dropout(set.dropout_enable), m_dropout_enabled(false),
        m_dropout_rate(set.dropout_rate)
    {
        m_weight = new double[m_current_d * m_prev_d];
        m_bias = new double[m_current_d];

        if (m_uses_dropout)
            m_dropout_coeff = new double[m_current_d];

        /* weight and bias initializaion */
        std::random_device rd;
        std::mt19937 rgen(rd());
        std::normal_distribution<double> dist_w(0.0, std::sqrt(1.0 / m_current_d));
        std::normal_distribution<double> dist_b(0.0, 1.0);

        for (size_t i = 0; i < m_current_d * m_prev_d; i++)
            m_weight[i] = dist_w(rgen);
        for (size_t i = 0; i < m_current_d; i++)
            m_bias[i] = dist_b(rgen);
    }

    SigmoidLayer::~SigmoidLayer()
    {
        if (m_uses_dropout)
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

        for (int i=0; i<m_train_num; i++)
        {
            mul_mat_vec(m_weight, prev_a + (i*m_prev_d), cur_z + (i*m_current_d),
                    m_current_d, m_prev_d);
            add_vec(cur_z + (i*m_current_d), m_bias,
                    cur_a + (i*m_current_d), m_current_d);
        }
        apply_vec(cur_a, cur_a, m_current_d * m_train_num, ActivationFuncs::f_sigmoid);

        if (m_uses_dropout)
        {
            // dropout is applied only to the activation value of the dropout-enabled layer
            // in the forwarding phase
            if (m_dropout_enabled)
            {
                // if this layer uses dropout, select neurons which does not participate in training
                // by setting their activation value to zero.
                std::random_device rd;
                std::mt19937 rgen(rd());
                std::uniform_real_distribution<> dis(0, 1);
                const double dropout_rate = m_dropout_rate;

                apply_vec(m_dropout_coeff, m_dropout_coeff, m_current_d,
                        [dropout_rate, &rgen, &dis](double in) -> double {
                            return (dis(rgen) <= dropout_rate) ? 1:0;
                        });
                for (size_t m = 0; m < m_train_num; m++)
                {
                    pmul_vec(cur_a + (m_current_d*m), m_dropout_coeff,
                            cur_a + (m_current_d*m), m_current_d);
                }
            }
            else
            {
                // otherwise, just multiply the dropout rate to every activation value
                const_mul_vec(cur_a, m_dropout_rate, m_train_num * m_current_d);
            }
        }
    }

    void SigmoidLayer::forward_gpu(const LayerData& prev, LayerData& current)
    {
        /* TODO */
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
        double *sprime_z = new double[m_prev_d * m_train_num];
        apply_vec(prev_z, sprime_z, m_prev_d * m_train_num, ActivationFuncs::f_sigmoid_prime);

        double *temp_w = new double[m_prev_d * m_current_d];
        transpose_mat(m_weight, temp_w, m_current_d, m_prev_d);

        for (int i=0; i<m_train_num; i++)
        {
            mul_mat_vec(temp_w, cur_e + (i*m_current_d), prev_e + (i*m_prev_d), m_prev_d, m_current_d);
        }
        pmul_vec(prev_e, sprime_z, prev_e, m_prev_d * m_train_num);

        /* calculate delta_b and update current bias */
        double *delta_b = new double[m_current_d];

        sum_vec(cur_e, delta_b, m_current_d, m_train_num);
        apply_vec(delta_b, delta_b, m_current_d, [train_num, learn_rate](double in) -> double {
            return -in*learn_rate/train_num;
        });
        add_vec(m_bias, delta_b, m_bias, m_current_d);

        /* calculate delta_w and update current weight */
        double *delta_w = new double[m_prev_d * m_current_d];
        memset(delta_w, 0, sizeof(double) * m_prev_d * m_current_d);

        for (int i=0; i<m_train_num; i++)
        {
            vec_outer_prod(cur_e, prev_a, temp_w, m_current_d, m_prev_d);
            add_vec(delta_w, temp_w, delta_w, m_prev_d * m_current_d);
        }
        apply_vec(delta_w, delta_w, m_prev_d * m_current_d, [train_num, learn_rate](double in) -> double {
            return -in*learn_rate/train_num;
        });
        add_vec(m_weight, delta_w, m_weight, m_prev_d * m_current_d);

        delete [] delta_w;
        delete [] delta_b;
        delete [] temp_w;
        delete [] sprime_z;
    }

    void SigmoidLayer::backward_gpu(LayerData& prev, LayerData& current)
    {
        /* TODO */
    }

    std::unique_ptr<LayerData> SigmoidLayer::createLayerData(size_t train_num)
    {
        return std::make_unique<LayerData>(
            train_num,
            m_current_d
        );
    }

    void SigmoidLayer::importLayer(const Json::Value& coeffs)
    {
        size_t neurons = coeffs["neurons"].asUInt();
        if (neurons != m_current_d)
            throw Json::LogicError("invalid 'neurons' in layer_data");

        auto weight_lists = coeffs["weight"];
        if (weight_lists.size() != m_current_d)
            throw Json::LogicError("invalid number of weight values");
        for (int i = 0; i < m_current_d; i++)
        {
            auto weights = weight_lists[i];
            if (weights.size() != m_prev_d)
                throw Json::LogicError("invalid number of weight values");

            for (int j = 0; j < m_prev_d; j++)
                m_weight[i * m_prev_d + j] = weights[j].asDouble();
        }

        auto biases = coeffs["bias"];
        if (biases.size() != m_current_d)
            throw Json::LogicError("invalid number of bias values");
        for (int i = 0; i < m_current_d; i++)
            m_bias[i] = biases[i].asDouble();
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
