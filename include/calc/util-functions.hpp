#ifndef __UTIL_FUNCTIONS_HPP
#define __UTIL_FUNCTIONS_HPP

#include <functional>

namespace NeuralNet
{
	class ActivationFuncs
	{
	public:
		static const std::function<double(double)> f_sigmoid;
		static const std::function<double(double)> f_sigmoid_prime;
		
		static const std::function<double(double)> f_relu;
		static const std::function<double(double)> f_relu_prime;
	};
}

#endif // __UTIL_FUNCTIONS_HPP