#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <cmath>
#include "netpbm/pm.h"
#include "utils/make_unique.hpp"
#include "utils/cl_exception.hpp"
#include "json/json.h"
#include "image/image.hpp"
#include "image/image_util.hpp"
#include "network.hpp"

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/cl.hpp"
#include "cl_context.hpp"

#include "face_finder.hpp"

int main(int argc, char* argv[])
{
    pm_init(argv[0], 0);

    NeuralNet::FaceFinder finder("../net_set/search_face.json");

    std::vector< std::string > img_str{
        "300imgs/n1.bmp", "300imgs/n2.bmp", "300imgs/n3.bmp",
        "300imgs/f1.bmp", "300imgs/f2.bmp", "300imgs/f3.bmp",
        "300imgs/test3.bmp", "300imgs/img_198.bmp"
    };

    for (auto& path: img_str)
    {
        auto value = finder.evaluate(path);
        std::cout << "value = (" << value[0] << "," << value[1] << ")" <<
            std::endl;
    }
}
