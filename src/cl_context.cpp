#include "cl_context.hpp"
#include <iostream>

namespace NeuralNet
{
    CLContext::CLContext()
    {
        cl_int err = CL_SUCCESS;

        std::vector<cl::Platform> platforms;
        err = cl::Platform::get(&platforms);
        if (err != CL_SUCCESS || platforms.size() == 0)
        {
            throw CLContextException(std::string("error at Platform::get, code: ")
                    + std::to_string(err));
        }

        std::cout << "list of platforms:" << std::endl;
        for (auto& platform: platforms)
        {
            std::cout << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        }
        
        m_platform = platforms[m_default_platf_num];
        std::cout << "using platform: " << m_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

        std::vector<cl::Device> devices;
        err = m_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if (err != CL_SUCCESS || devices.size() == 0)
        {
            throw CLContextException(std::string("error at Platform::getDevices, code: ")
                    + std::to_string(err));
        }

        std::cout << "list of devices:" << std::endl;
        for (auto& device: devices)
        {
            std::cout << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        }

        m_device = devices[m_default_dev_num];
        std::cout << "using device: " << m_device.getInfo<CL_DEVICE_NAME>() << std::endl;

        auto sources = loadSources();
     
        context = cl::Context(m_device);
        program = cl::Program(context, sources);

        try
        {
            program.build({m_device});
        }
        catch (cl::Error e)
        {
            std::cerr << "Error at " << e.what() << std::endl;
            throw CLContextException(std::string("error at Program::build: ")
                    + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device));
        }

        queue = cl::CommandQueue(context,m_device);
    }

    std::string CLContext::loadSources()
    {
        std::vector<std::string> src_file_list = {
            "../cl_src/helper.cl",
            "../cl_src/sigmoid.cl",
        //  "../cl_src/conv.cl",
        //  "../cl_src/maxpool.cl"    
        };
        std::string sources;

        for (auto& src_file_name: src_file_list)
        {
            std::ifstream ifs(src_file_name);
            if (!ifs)
            {
                throw CLContextException(std::string("error reading file: ") + src_file_name);
            }

            std::string source_str;
            while (!ifs.eof())
            {
                std::string line;
                std::getline(ifs, line);
                source_str += (line + '\n');
            }
            sources += source_str;
        }

        return sources;
    }

    CLContext::~CLContext()
    {
    }
}
