#ifndef __UTIL_FUNCTIONS_HPP
#define __UTIL_FUNCTIONS_HPP

#include <functional>

namespace NeuralNet
{
    class ActivationFuncs
    {
    public:
        static const std::function<float(float)> f_sigmoid;
        static const std::function<float(float)> f_sigmoid_prime;

        static const std::function<float(float)> f_relu;
        static const std::function<float(float)> f_relu_prime;
    };
}

#endif // __UTIL_FUNCTIONS_HPP