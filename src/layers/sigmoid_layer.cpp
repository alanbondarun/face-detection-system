#include "layers/sigmoid_layer.hpp"
#include "calc/calc-cpu.hpp"
#include <cstring>

namespace NeuralNet
{
	const std::function<double(double)> SigmoidLayer::f_sigmoid = [](double in) {
		return 1.0 / (1.0 + std::exp(-in));
	};
	const std::function<double(double)> SigmoidLayer::f_sigmoid_prime = [](double in) {
		double s_in = SigmoidLayer::f_sigmoid(in);
		return s_in * (1.0 - s_in);
	};
	const double SigmoidLayer::eta = 0.1;
	
	void SigmoidLayer::forward_cpu(const LayerData& prev, LayerData& current)
	{		
		auto prev_d = prev.getNodeSize();
		auto cur_d = current.getNodeSize();
		auto train_num = current.getTrainNum();
		auto prev_a = prev.get(LayerData::DataIndex::ACTIVATION);
		auto cur_w = current.get(LayerData::DataIndex::WEIGHT);
		auto cur_b = current.get(LayerData::DataIndex::BIAS);
		auto cur_z = current.get(LayerData::DataIndex::INTER_VALUE);
		auto cur_a = current.get(LayerData::DataIndex::ACTIVATION);
		
		for (int i=0; i<train_num; i++)
		{	
			mul_mat_vec(cur_w, prev_a + (i*prev_d), cur_z + (i*cur_d), cur_d, prev_d);
			add_vec(cur_z + (i*cur_d), cur_b + (i*cur_d), cur_a + (i*cur_d), cur_d);
		}
		apply_vec(cur_a, cur_d * train_num, f_sigmoid);
	}
	
	void SigmoidLayer::forward_gpu(const LayerData& prev, LayerData& current)
	{
		/* TODO */
	}
	
	void SigmoidLayer::backward_cpu(LayerData& prev, LayerData& current)
	{
		auto prev_d = prev.getNodeSize();
		auto cur_d = current.getNodeSize();
		auto train_num = current.getTrainNum();
		auto prev_e = prev.get(LayerData::DataIndex::ERROR);
		auto prev_z = prev.get(LayerData::DataIndex::INTER_VALUE);
		auto prev_a = prev.get(LayerData::DataIndex::ACTIVATION);
		auto cur_w = current.get(LayerData::DataIndex::WEIGHT);
		auto cur_b = current.get(LayerData::DataIndex::BIAS);
		auto cur_e = current.get(LayerData::DataIndex::ERROR);
		auto cur_z = current.get(LayerData::DataIndex::INTER_VALUE);
		auto cur_a = current.get(LayerData::DataIndex::ACTIVATION);
		
		/* calculate error value for previous layer */
		double *sprime_z = new double[prev_d * train_num];
		memcpy(sprime_z, prev_z, sizeof(double) * prev_d * train_num);
		apply_vec(sprime_z, prev_d * train_num, f_sigmoid_prime);
		
		double *temp_w = new double[prev_d * cur_d];
		transpose_mat(cur_w, temp_w, cur_d, prev_d);
		
		for (int i=0; i<train_num; i++)
		{
			mul_mat_vec(temp_w, cur_e + (i*cur_d), prev_e + (i*prev_d), prev_d, cur_d);
		}
		pmul_vec(prev_e, sprime_z, prev_e, prev_d * train_num);
		
		/* calculate delta_b and update current bias */
		double *delta_b = new double[cur_d];
		
		sum_vec(cur_e, delta_b, cur_d, train_num);
		apply_vec(delta_b, cur_d, [eta, train_num](double in) -> double {
			return -in*eta/train_num;
		});
		add_vec(cur_b, delta_b, cur_b, cur_d);
		
		/* calculate delta_w and update current weight */
		double *delta_w = new double[prev_d * cur_d];
		memset(delta_w, 0, sizeof(double) * prev_d * cur_d);
		
		for (int i=0; i<train_num; i++)
		{
			vec_outer_prod(cur_e, prev_a, temp_w, cur_d, prev_d);
			add_vec(delta_w, temp_w, delta_w, prev_d * cur_d);
		}
		apply_vec(delta_w, prev_d * cur_d, [eta, train_num](double in) -> double {
			return -in*eta/train_num;
		});
		add_vec(cur_w, delta_w, cur_w, prev_d * cur_d);
		
		delete [] delta_w;
		delete [] delta_b;
		delete [] temp_w;
		delete [] sprime_z;
	}
	
	void SigmoidLayer::backward_gpu(LayerData& prev, LayerData& current)
	{
		/* TODO */
	}
}
