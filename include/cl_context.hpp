#ifndef __CL_CONTEXT_HPP
#define __CL_CONTEXT_HPP

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/cl.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

namespace NeuralNet
{
    class CLContext
    {
    private:
        cl::CommandQueue queue;
        cl::Program program;
        cl::Context context;

        // emits cl::Error or CLContext::Exception if any
        CLContext();

    public:
        ~CLContext();

        // customized exception class
        class CLContextException: public std::exception
        {
        public:
            CLContextException(const std::string& msg)
                : m_msg(std::string("Error at CLContext: ") + msg) {}
            virtual ~CLContextException() {}
            virtual const char* what() const noexcept { return m_msg.c_str(); }

        private:
            const std::string m_msg;
        };

        // do not allow copying & moving
        CLContext(const CLContext&) = delete;
        CLContext& operator=(const CLContext&) = delete;
        CLContext(CLContext&&) noexcept = delete;
        CLContext& operator=(CLContext&&) noexcept = delete;

        static CLContext& getInstance()
        {
            static CLContext in;
            return in;
        }

        cl::CommandQueue getCommandQueue() { return queue; }
        cl::Program getProgram() { return program; }
        cl::Context getContext() { return context; }

    private:
        cl::Platform m_platform;
        cl::Device m_device;

        std::string loadSources();

        const size_t m_default_platf_num = 0;
        const size_t m_default_dev_num = 0;
    };
}

#endif // __CL_CONTEXT_HPP
