#include "image/image.hpp"
#include "netpbm/ppm.h"
#include <iostream>
#include <utility>

int main(int argc, char* argv[])
{
    pm_init(argv[0], 0);

    auto img1 = NeuralNet::loadPPMImage("00227_940128_fa.ppm");
    if (!img1)
    {
        printf("img1 error\n");
        return 0;
    }

    auto img2 = NeuralNet::loadPGMImage("BioID_0277.pgm");
    if (!img2)
    {
        printf("img2 error\n");
        return 0;
    }

    auto img5 = NeuralNet::loadBitmapImage("37-big.bmp");
    if (!img5)
    {
        printf("img5 error\n");
        return 0;
    }

    auto img3 = NeuralNet::fitImageTo(img1, 32, 32);
    auto img4 = NeuralNet::fitImageTo(img2, 32, 32);
    auto img6 = NeuralNet::grayscaleImage(NeuralNet::fitImageTo(img5, 32, 32));

    if (!NeuralNet::saveAsPPM(img3, "image3.ppm"))
    {
        printf("error saving image3.ppm\n");
    }

    if (!NeuralNet::saveAsPPM(img4, "image4.ppm"))
    {
        printf("error saving image4.ppm\n");
    }

    if (!NeuralNet::saveAsPPM(img6, "37.ppm"))
    {
        printf("error saving 37.ppm\n");
    }
}
