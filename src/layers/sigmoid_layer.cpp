#include "layers/sigmoid_layer.hpp"
#include "calc/calc-cpu.hpp"

namespace NeuralNet
{
	void SigmoidLayer::forward_cpu(const LayerData& prev, LayerData& current)
	{
		size_t prev_d = prev.getNum();
		size_t cur_d = current.getNum();
		double *prev_a = prev.get(LayerData::DataIndex::ACTIVATION);
		double *cur_w = current.get(LayerData::DataIndex::WEIGHT);
		double *cur_b = current.get(LayerData::DataIndex::BIAS);
		double *cur_z = current.get(LayerData::DataIndex::INTER_VALUE);
		double *cur_a = current.get(LayerData::DataIndex::ACTIVATION);
		
		mul_mat_vec(cur_w, prev_a, cur_z, cur_d, prev_d);
		add_vec(cur_z, cur_b, cur_a, cur_d);
		apply_vec(cur_a, cur_d, [](double in) -> double {
		    return 1.0 / (1.0 + std::exp(-in));
		});
	}
	
	void SigmoidLayer::forward_gpu(const LayerData& prev, LayerData& current)
	{
		/* TODO */
	}
	
	void SigmoidLayer::backward_cpu(LayerData& prev, LayerData& current)
	{
		size_t prev_d = prev.getNum();
		size_t cur_d = current.getNum();
		double *prev_e = prev.get(LayerData::DataIndex::ERROR);
		double *cur_w = current.get(LayerData::DataIndex::WEIGHT);
		double *cur_e = current.get(LayerData::DataIndex::ERROR);
		double *cur_z = current.get(LayerData::DataIndex::INTER_VALUE);
		double *cur_a = current.get(LayerData::DataIndex::ACTIVATION);
		double *transpose_w = new double[prev_d * cur_d];
		
		transpose_mat(cur_w, transpose_w, cur_d, prev_d);
		
		
		
		delete [] transpose_w;
	}
	
	void SigmoidLayer::backward_gpu(LayerData& prev, LayerData& current)
	{
		/* TODO */
	}
}
