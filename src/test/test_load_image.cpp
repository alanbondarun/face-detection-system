#include "image/image.hpp"
#include "netpbm/ppm.h"
#include <iostream>
#include <utility>

int main(int argc, char* argv[])
{
    pm_init(argv[0], 0);

    auto img1 = NeuralNet::loadBitmapImage("eval-image/2.bmp");
    if (!img1)
    {
        printf("img1 error\n");
        return 0;
    }

    auto img2 = NeuralNet::loadBitmapImage("eval-image/6.bmp");
    if (!img2)
    {
        printf("img2 error\n");
        return 0;
    }

    auto img3 = NeuralNet::loadBitmapImage("eval-image/11.bmp");
    if (!img3)
    {
        printf("img3 error\n");
        return 0;
    }

    auto img1_res = NeuralNet::leastSquarePatch(
            NeuralNet::equalizePatch(NeuralNet::grayscaleImage(img1)));
    auto img2_res = NeuralNet::leastSquarePatch(
            NeuralNet::equalizePatch(NeuralNet::grayscaleImage(img2)));
    auto img3_res = NeuralNet::leastSquarePatch(
            NeuralNet::equalizePatch(NeuralNet::grayscaleImage(img3)));

    if (!NeuralNet::saveAsPPM(img1_res, "image1.ppm"))
    {
        printf("error saving image1.ppm\n");
    }

    if (!NeuralNet::saveAsPPM(img2_res, "image2.ppm"))
    {
        printf("error saving image2.ppm\n");
    }

    if (!NeuralNet::saveAsPPM(img3_res, "image3.ppm"))
    {
        printf("error saving image3.ppm\n");
    }
}
