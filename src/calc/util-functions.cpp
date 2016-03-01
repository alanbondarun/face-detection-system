#include "calc/util-functions.hpp"
#include <cmath>

namespace NeuralNet
{
    const std::function<double(double)> ActivationFuncs::f_sigmoid = [](double in) {
            return 1.0 / (1.0 + std::exp(-in));
        };
    const std::function<double(double)> ActivationFuncs::f_sigmoid_prime = [](double in) {
        double s_in = f_sigmoid(in);
        return s_in * (1.0 - s_in);
    };

    const std::function<double(double)> ActivationFuncs::f_relu = [](double in) {
        return std::max(0.0, in);
    };
    const std::function<double(double)> ActivationFuncs::f_relu_prime = [](double in) {
        if (in <= 0.0)
            return 0.0;
        return 1.0;
    };
}