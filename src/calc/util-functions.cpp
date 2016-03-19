#include "calc/util-functions.hpp"
#include <cmath>

namespace NeuralNet
{
    const std::function<float(float)> ActivationFuncs::f_sigmoid = [](float in) {
            return 1.0 / (1.0 + std::exp(-in));
        };
    const std::function<float(float)> ActivationFuncs::f_sigmoid_prime = [](float in) {
        float s_in = f_sigmoid(in);
        return s_in * (1.0 - s_in);
    };

    const std::function<float(float)> ActivationFuncs::f_relu = [](float in) {
        return std::max(static_cast<float>(0.0), in);
    };
    const std::function<float(float)> ActivationFuncs::f_relu_prime = [](float in) {
        if (in <= 0.0)
            return 0.0;
        return 1.0;
    };
}
