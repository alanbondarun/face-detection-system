#include "utils/cl_exception.hpp"

namespace NeuralNet
{
    void printError(cl_int error_code, std::string error_str)
    {
        if (error_code != CL_SUCCESS)
        {
            std::cout << error_str << ", code = " << error_code << std::endl;
        }
    }
}
