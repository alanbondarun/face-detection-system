#ifndef __UTIL_FUNCTIONS_HPP
#define __UTIL_FUNCTIONS_HPP

#include <functional>
#include <cmath>

namespace NeuralNet
{
	const std::function<double(double)> f_sigmoid = [](double in) {
		return 1.0 / (1.0 + std::exp(-in));
	};
	const std::function<double(double)> f_sigmoid_prime = [](double in) {
		double s_in = f_sigmoid(in);
		return s_in * (1.0 - s_in);
	};
	
	const std::function<double(double)> f_relu = [](double in) {
		return std::max(0.0, in);
	};
	const std::function<double(double)> f_relu_prime = [](double in) {
		if (in <= 0.0)
			return 0.0;
		return 1.0;
	};
}

#endif // __UTIL_FUNCTIONS_HPP