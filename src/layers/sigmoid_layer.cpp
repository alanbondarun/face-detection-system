#include "layers/sigmoid_layer.hpp"
#include "calc/calc-cpu.hpp"

namespace NeuralNet
{
	const std::function<double(double)> SigmoidLayer::f_sigmoid = [](double in) {
		return 1.0 / (1.0 + std::exp(-in));
	};
	const std::function<double(double)> f_sigmoid_prime = [](double in) {
		double s_in = SigmoidLayer::f_sigmoid(in);
		return s_in * (1.0 - s_in);
	};
	
	void SigmoidLayer::forward_cpu(const LayerData& prev, LayerData& current)
	{
		auto prev_d = prev.getNodeSize();
		auto cur_d = current.getNodeSize();
		auto prev_a = prev.get(LayerData::DataIndex::ACTIVATION);
		auto cur_w = current.get(LayerData::DataIndex::WEIGHT);
		auto cur_b = current.get(LayerData::DataIndex::BIAS);
		auto cur_z = current.get(LayerData::DataIndex::INTER_VALUE);
		auto cur_a = current.get(LayerData::DataIndex::ACTIVATION);
		
		mul_mat_vec(cur_w, prev_a, cur_z, cur_d, prev_d);
		add_vec(cur_z, cur_b, cur_a, cur_d);
		apply_vec(cur_a, cur_d, f_sigmoid);
	}
	
	void SigmoidLayer::forward_gpu(const LayerData& prev, LayerData& current)
	{
		/* TODO */
	}
	
	void SigmoidLayer::backward_cpu(LayerData& prev, LayerData& current)
	{
		auto prev_d = prev.getNodeSize();
		auto cur_d = current.getNodeSize();
		auto prev_e = prev.get(LayerData::DataIndex::ERROR);
		auto cur_w = current.get(LayerData::DataIndex::WEIGHT);
		auto cur_e = current.get(LayerData::DataIndex::ERROR);
		auto cur_z = current.get(LayerData::DataIndex::INTER_VALUE);
		auto cur_a = current.get(LayerData::DataIndex::ACTIVATION);
		
		double *transpose_w = new double[prev_d * cur_d];
		
		/* TODO */
		
		transpose_mat(cur_w, transpose_w, cur_d, prev_d);
		
		
		
		delete [] transpose_w;
	}
	
	void SigmoidLayer::backward_gpu(LayerData& prev, LayerData& current)
	{
		/* TODO */
	}
}
