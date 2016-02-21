#include "layers/sigmoid_layer.hpp"
#include "calc/calc-cpu.hpp"
#include "calc/util-functions.hpp"
#include "utils/make_unique.hpp"
#include <cstring>

namespace NeuralNet
{
	const double SigmoidLayer::eta = 0.1;
	
	SigmoidLayer::SigmoidLayer(size_t prev_neurons, size_t current_neurons, size_t train_num)
		: m_prev_d(prev_neurons), m_current_d(current_neurons), m_train_num(train_num)
	{
		m_weight = new double[current_neurons * prev_neurons];
		m_bias = new double[current_neurons];
		
		/* TODO: value init? */
	}
	
	SigmoidLayer::~SigmoidLayer()
	{
		delete [] m_bias;
		delete [] m_weight;
	}
	
	void SigmoidLayer::forward_cpu(const LayerData& prev, LayerData& current)
	{
		/* TODO: data correctness check? */
		
		auto prev_a = prev.get(LayerData::DataIndex::ACTIVATION);
		auto cur_z = current.get(LayerData::DataIndex::INTER_VALUE);
		auto cur_a = current.get(LayerData::DataIndex::ACTIVATION);
		
		for (int i=0; i<m_train_num; i++)
		{	
			mul_mat_vec(m_weight, prev_a + (i*m_prev_d), cur_z + (i*m_current_d),
					m_current_d, m_prev_d);
			add_vec(cur_z + (i*m_current_d), m_bias + (i*m_current_d),
					cur_a + (i*m_current_d), m_current_d);
		}
		apply_vec(cur_a, cur_a, m_current_d * m_train_num, f_sigmoid);
	}
	
	void SigmoidLayer::forward_gpu(const LayerData& prev, LayerData& current)
	{
		/* TODO */
	}
	
	void SigmoidLayer::backward_cpu(LayerData& prev, LayerData& current)
	{
		/* TODO: data correctness check? */
		const auto train_num = m_train_num;
		auto prev_e = prev.get(LayerData::DataIndex::ERROR);
		auto prev_z = prev.get(LayerData::DataIndex::INTER_VALUE);
		auto prev_a = prev.get(LayerData::DataIndex::ACTIVATION);
		auto cur_e = current.get(LayerData::DataIndex::ERROR);
		
		/* calculate error value for previous layer */
		double *sprime_z = new double[m_prev_d * m_train_num];
		apply_vec(prev_z, sprime_z, m_prev_d * m_train_num, f_sigmoid_prime);
		
		double *temp_w = new double[m_prev_d * m_current_d];
		transpose_mat(m_weight, temp_w, m_current_d, m_prev_d);
		
		for (int i=0; i<m_train_num; i++)
		{
			mul_mat_vec(temp_w, cur_e + (i*m_current_d), prev_e + (i*m_prev_d), m_prev_d, m_current_d);
		}
		pmul_vec(prev_e, sprime_z, prev_e, m_prev_d * m_train_num);
		
		/* calculate delta_b and update current bias */
		double *delta_b = new double[m_current_d];
		
		sum_vec_preserve(cur_e, delta_b, m_current_d, m_train_num);
		apply_vec(delta_b, delta_b, m_current_d, [train_num](double in) -> double {
			return -in*eta/train_num;
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
		apply_vec(delta_w, delta_w, m_prev_d * m_current_d, [train_num](double in) -> double {
			return -in*eta/train_num;
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
	
	std::unique_ptr<LayerData> SigmoidLayer::createLayerData()
	{
		return std::make_unique<LayerData>(
			m_train_num,
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
