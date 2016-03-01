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

    auto img5 = NeuralNet::loadPPMImage("qq.ppm");
    if (!img5)
    {
        printf("img5 error\n");
        return 0;
    }

    auto img3 = NeuralNet::fitImageTo(img1, 32, 32);
    auto img4 = NeuralNet::fitImageTo(img2, 32, 32);
    auto img6 = NeuralNet::fitImageTo(img5, 32, 32);

    FILE* fp_img3 = fopen("image3.ppm", "wb");
    FILE* fp_img4 = fopen("image4.ppm", "wb");
    FILE* fp_img6 = fopen("image6.ppm", "wb");
    pixval maxVal = 255;

    pixel** px_img3 = ppm_allocarray(img3->getWidth(), img3->getHeight());
    for (size_t h = 0; h < img3->getHeight(); h++)
    {
        for (size_t w = 0; w < img3->getWidth(); w++)
        {
            px_img3[h][w].r = static_cast<pixval>((img3->getValues(0))[h * img3->getWidth() + w]
                    * maxVal);
            px_img3[h][w].g = static_cast<pixval>((img3->getValues(1))[h * img3->getWidth() + w]
                    * maxVal);
            px_img3[h][w].b = static_cast<pixval>((img3->getValues(2))[h * img3->getWidth() + w]
                    * maxVal);
        }
    }

    pixel** px_img4 = ppm_allocarray(img4->getWidth(), img4->getHeight());
    for (size_t h = 0; h < img4->getHeight(); h++)
    {
        for (size_t w = 0; w < img4->getWidth(); w++)
        {
            px_img4[h][w].r = static_cast<pixval>((img4->getValues(0))[h * img4->getWidth() + w]
                    * maxVal);
            px_img4[h][w].g = static_cast<pixval>((img4->getValues(1))[h * img4->getWidth() + w]
                    * maxVal);
            px_img4[h][w].b = static_cast<pixval>((img4->getValues(2))[h * img4->getWidth() + w]
                    * maxVal);
        }
    }

    pixel** px_img6 = ppm_allocarray(img6->getWidth(), img6->getHeight());
    for (size_t h = 0; h < img6->getHeight(); h++)
    {
        for (size_t w = 0; w < img6->getWidth(); w++)
        {
            px_img6[h][w].r = static_cast<pixval>((img6->getValues(0))[h * img6->getWidth() + w]
                    * maxVal);
            px_img6[h][w].g = static_cast<pixval>((img6->getValues(1))[h * img6->getWidth() + w]
                    * maxVal);
            px_img6[h][w].b = static_cast<pixval>((img6->getValues(2))[h * img6->getWidth() + w]
                    * maxVal);
        }
    }

    ppm_writeppm(fp_img3, px_img3, img3->getWidth(), img3->getHeight(), maxVal, 0);
    ppm_writeppm(fp_img4, px_img4, img4->getWidth(), img4->getHeight(), maxVal, 0);
    ppm_writeppm(fp_img6, px_img6, img6->getWidth(), img6->getHeight(), maxVal, 0);

    ppm_freearray(px_img3, img3->getHeight());
    ppm_freearray(px_img4, img4->getHeight());
    ppm_freearray(px_img6, img6->getHeight());

    fclose(fp_img3);
    fclose(fp_img4);
    fclose(fp_img6);
}
