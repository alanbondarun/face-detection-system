#ifndef __CL_EXCEPTION_HPP
#define __CL_EXCEPTION_HPP

#include <iostream>
#include <string>
#include "CL/cl.hpp"

namespace NeuralNet
{
    void printError(cl_int error_code, std::string error_str);
}

#endif // __CL_EXCEPTION_HPP
