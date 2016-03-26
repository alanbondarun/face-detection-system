#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <cmath>
#include <cstdio>
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

void analyzeResult(const std::vector< std::vector<float> >& values)
{
    int maxn=0, maxs=0;
    for (auto& val: values)
    {
        maxn = std::max(maxn, static_cast<int>(val[0]));
        maxs = std::max(maxs, static_cast<int>(val[1]));
    }

    std::vector<int> nums(maxn + 31), scores(maxs + 4);
    float a=0, b=0;
    for (auto& val: values)
    {
        a += val[0];
        b += val[1];

        nums[static_cast<size_t>(val[0])]++;
        scores[static_cast<size_t>(val[1])]++;
    }

    std::cout << "### avg positive patch: " << (a/values.size()) <<
        ", avg score: " << (b/values.size()) << std::endl;
    std::cout << "### nums" << std::endl;
    for (size_t i=0; i<nums.size(); i += 10)
    {
        printf("%3d ", static_cast<int>(i));

        int kk = 0;
        for (size_t j=i; j < std::min(i+10, nums.size()); j++)
        {
            kk += nums[j];
        }
        for (int j=0; j<kk/4; j++)
        {
            printf("#");
        }
        printf("\n");
    }
    std::cout << "\n### scores\n";
    for (size_t i=0; i<scores.size(); i++)
    {
        printf("%3d ", static_cast<int>(i));
        for (int j=0; j<scores[i]/2; j++)
        {
            printf("#");
        }
        printf("\n");
    }
}

void face(NeuralNet::FaceFinder& finder)
{
    const int num_pics = 200;
    std::vector< std::vector<float> > values;
    for (int i=1; i<=num_pics; i++)
    {
        std::ostringstream oss;
        oss << "../../fddb-images/img_" << i << ".bmp";

        auto value = finder.evaluate(oss.str());
        values.push_back(value);
    }
    std::cout << "### FACE" << std::endl;
    analyzeResult(values);
}

void nonface(NeuralNet::FaceFinder& finder)
{
    std::vector< std::vector<float> > values;
    for (int i=6001; i<=6400; i++)
    {
        std::ostringstream oss;
        oss << "300tests/img_" << i << ".bmp";

        auto value = finder.evaluate(oss.str());
        values.push_back(value);
    }

    std::cout << "### NONFACE" << std::endl;
    analyzeResult(values);
}

int main(int argc, char* argv[])
{
    pm_init(argv[0], 0);

    NeuralNet::FaceFinder finder("../net_set/search_face.json");
    face(finder);
    nonface(finder);
}
