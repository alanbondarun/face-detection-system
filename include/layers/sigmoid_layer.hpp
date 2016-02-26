#ifndef __SIGMOID_LAYER_HPP
#define __SIGMOID_LAYER_HPP

#include "layers/layer.hpp"
#include "json/json.h"
#include <functional>

namespace NeuralNet
{
	class SigmoidLayer: public Layer
	{
    public:
		struct Setting
		{
			size_t prev_neurons;
			size_t current_neurons;
			size_t train_num;
			double learn_rate;
			double dropout_rate;
			bool dropout_enable;
		};
		SigmoidLayer(const Setting& set);
        virtual ~SigmoidLayer();

		virtual void forward_cpu(const LayerData& prev, LayerData& current);
		virtual void forward_gpu(const LayerData& prev, LayerData& current);
		virtual void backward_cpu(LayerData& prev, LayerData& current);
		virtual void backward_gpu(LayerData& prev, LayerData& current);
		
		virtual std::unique_ptr<LayerData> createLayerData();

		virtual void importLayer(const Json::Value& coeffs);
		virtual Json::Value exportLayer();

		virtual std::string what() { return "sigmoid"; }

		void setDropout(bool enable) { m_dropout_enabled = enable; }

		static const std::function<double(double)> f_sigmoid;
		static const std::function<double(double)> f_sigmoid_prime;
		
	private:
		const size_t m_prev_d, m_current_d, m_train_num;
		const double m_learn_rate;

		const bool m_uses_dropout;
		bool m_dropout_enabled;
		const double m_dropout_rate;

		double *m_weight;
		double *m_bias;
		double *m_dropout_coeff;
	};
}

#endif // __SIGMOID_LAYER_HPP
