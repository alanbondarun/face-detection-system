#include <iostream>
#include <fstream>
#include <memory>
#include <vector>

#include <unistd.h>

#include "utils/make_unique.hpp"
#include "netpbm/pm.h"
#include "face_finder.hpp"

int main(int argc, char* argv[])
{
    pm_init(argv[0], 0);

    auto finder = std::make_unique<NeuralNet::FaceFinder>(
            "../net_set/search_face.json");

    while (true)
    {
        std::ofstream eval_file("dpms.txt");

        try
        {
            auto value = finder->evaluate("img_3.bmp");
            if (value[0] >= 54)
            {
                eval_file << "1";
            }
            else
            {
                eval_file << "0";
            }
        }
        catch (NeuralNet::FaceFinder::FaceFinderException& e)
        {
            eval_file << "0";
        }

        eval_file.flush();
        usleep(200000);
    }

    return 0;
}
