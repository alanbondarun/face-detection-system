#include "image/image.hpp"
#include "utils/make_unique.hpp"
#include "netpbm/ppm.h"
#include "netpbm/pgm.h"
#include <stdio.h>
#include <cstdint>
#include <utility>
#include <vector>

namespace NeuralNet
{
    Image::Image(size_t _width, size_t _height, size_t _channels)
        : width(_width), height(_height), channel_num(_channels)
    {
        value = new double*[channel_num];
        for (size_t i = 0; i < channel_num; i++)
        {
            value[i] = new double[width * height];
        }
    }

    Image::~Image()
    {
        for (size_t i = 0; i < channel_num; i++)
        {
            delete [] value[i];
        }
        delete [] value;
    }

    std::unique_ptr<Image> loadBitmapImage(const char* filepath)
    {
        FILE* fp = fopen(filepath, "r");
        if (!fp)
        {
            printf("error loading file %s at loadBitmapImage()\n", filepath);
            return std::unique_ptr<Image>();
        }

        unsigned char info[54];
        if (fread(info, sizeof(unsigned char), 54, fp) < 54)
        {
            printf("error reading file %s at loadBitmapImage()\n", filepath);
            fclose(fp);
            return std::unique_ptr<Image>();
        }
        const uint32_t img_width = *(uint32_t*)(&info[18]);
        const uint32_t img_height = *(uint32_t*)(&info[22]);

        auto img_ptr = std::make_unique<Image>(img_width, img_height, 3);

        size_t row_padded = (img_width*3 + 3)&(~3);
        std::vector<unsigned char> raw_data(row_padded, 0);

        for (size_t h = 0; h < img_height; h++)
        {
            if (fread(raw_data.data(), sizeof(unsigned char), row_padded, fp) < row_padded)
            {
                printf("error reading file %s at loadBitmapImage()\n", filepath);
                fclose(fp);
                return std::unique_ptr<Image>();
            }
            for (size_t w = 0; w < img_width; w++)
            {
                (img_ptr->getValues(0))[(img_height - 1 - h) * img_width + w]
                        = raw_data[3*w + 2] / 255.0;
                (img_ptr->getValues(1))[(img_height - 1 - h) * img_width + w]
                        = raw_data[3*w + 1] / 255.0;
                (img_ptr->getValues(2))[(img_height - 1 - h) * img_width + w]
                        = raw_data[3*w] / 255.0;
            }
        }

        fclose(fp);
        return img_ptr;
    }

    std::unique_ptr<Image> loadBitmapImage(const std::string& filepath)
    {
        return loadBitmapImage(filepath.c_str());
    }

    std::unique_ptr<Image> shrinkImage(const std::unique_ptr<Image>& image, int w)
    {
        int orig_w = image->getWidth();
        int orig_h = image->getHeight();
        int h = (w * image->getHeight()) / image->getWidth();
        auto img_ptr = std::make_unique<Image>(w, h, image->getChannelNum());
        double* buffer = new double[(image->getWidth()+1) * (image->getHeight()+1)];

        for (size_t ch = 0; ch < image->getChannelNum(); ch++)
        {
            auto orig_data_ptr = image->getValues(ch);
            auto res_data_ptr = img_ptr->getValues(ch);

            // TODO: additional prefilter?

            // make a summed area table
            buffer[0] = 0;
            for (int i = 1; i <= orig_w; i++)
            {
                buffer[i] = 0;
            }
            for (int j = 1; j <= orig_h; j++)
            {
                buffer[j * orig_w] = 0;
            }
            for (int j = 1; j <= orig_h; j++)
            {
                for (int i = 1; i <= orig_w; i++)
                {
                    buffer[j*orig_w + i] = orig_data_ptr[(j-1)*orig_w + (i-1)]
                            + buffer[(j-1)*orig_w + i] + buffer[j*orig_w + (i-1)]
                            - buffer[(j-1)*orig_w + (i-1)];
                }
            }

            // average out the surrounding values
            int di = orig_w / w;
            int dj = orig_h / h;
            for (int j = 0; j < h; j++)
            {
                for (int i = 0; i < w; i++)
                {
                    res_data_ptr[j*w + i] = buffer[(j+1)*dj*orig_w + (i+1)*di]
                            - buffer[j*dj*orig_w + (i+1)*di]
                            - buffer[(j+1)*dj*orig_w + i*di]
                            + buffer[j*dj*orig_w + i*di];
                    res_data_ptr[j*w + i] /= static_cast<double>(di*dj);
                }
            }
        }

        delete [] buffer;
        return img_ptr;
    }

    std::unique_ptr<Image> cropImage(const std::unique_ptr<Image>& image, int x, int y, int w, int h)
    {
        auto img_ptr = std::make_unique<Image>(w, h, image->getChannelNum());
        for (size_t ch = 0; ch < image->getChannelNum(); ch++)
        {
            auto orig_data_ptr = image->getValues(ch);
            auto res_data_ptr = img_ptr->getValues(ch);

            for (int j = 0; j < h; j++)
            {
                for (int i = 0; i < w; i++)
                {
                    int ii = i + x;
                    int jj = j + y;
                    if (ii < image->getWidth() && jj < image->getHeight())
                    {
                        res_data_ptr[j*w + i] = orig_data_ptr[jj*image->getWidth() + ii];
                    }
                    else
                    {
                        res_data_ptr[j*w + i] = 0;
                    }
                }
            }
        }
        return img_ptr;
    }

    std::unique_ptr<Image> fitImageTo(const std::unique_ptr<Image>& image, int w, int h)
    {
        int pw = (h * image->getWidth()) / image->getHeight();
        int ph = (w * image->getHeight()) / image->getWidth();

        if (ph <= h)
        {
            auto middle_img = std::move(shrinkImage(image, pw));
            return cropImage(middle_img, pw/2 - w/2, 0, w, h);
        }
        auto middle_img = std::move(shrinkImage(image, w));
        return cropImage(middle_img, 0, ph/2 - h/2, w, h);
    }

    std::unique_ptr<Image> loadJPEGImage(const char* filepath)
    {
        /* TODO */
        return std::unique_ptr<Image>();
    }

    std::unique_ptr<Image> loadPPMImage(const char* filepath)
    {
        FILE* fp = fopen(filepath, "rb");
        if (!fp)
        {
            printf("error loading file %s at loadPPMImage()\n", filepath);
            return std::unique_ptr<Image>();
        }

        int img_w, img_h;
        pixval max_pval;
        pixel** pixels = ppm_readppm(fp, &img_w, &img_h, &max_pval);
        if (!pixels)
        {
            printf("error reading image from file %s at loadPPMImage()\n", filepath);
            fclose(fp);
            return std::unique_ptr<Image>();
        }

        auto img_ptr = std::make_unique<Image>(img_w, img_h, 3);

        for (size_t y = 0; y < img_h; y++)
        {
            for (size_t x = 0; x < img_w; x++)
            {
                (img_ptr->getValues(0))[y * img_w + x] = static_cast<double>(pixels[y][x].r)
                        / max_pval;
                (img_ptr->getValues(1))[y * img_w + x] = static_cast<double>(pixels[y][x].g)
                        / max_pval;
                (img_ptr->getValues(2))[y * img_w + x] = static_cast<double>(pixels[y][x].b)
                        / max_pval;
            }
        }

        ppm_freearray(pixels, img_h);
        fclose(fp);
        return img_ptr;
    }

    std::unique_ptr<Image> loadPGMImage(const char* filepath)
    {
        FILE* fp = fopen(filepath, "rb");
        if (!fp)
        {
            printf("error loading file %s at loadPGMImage()\n", filepath);
            return std::unique_ptr<Image>();
        }

        int img_w, img_h;
        gray max_pval;
        gray** pixels = pgm_readpgm(fp, &img_w, &img_h, &max_pval);
        if (!pixels)
        {
            printf("error reading image from file %s at loadPGMImage()\n", filepath);
            fclose(fp);
            return std::unique_ptr<Image>();
        }

        auto img_ptr = std::make_unique<Image>(img_w, img_h, 3);

        for (size_t y = 0; y < img_h; y++)
        {
            for (size_t x = 0; x < img_w; x++)
            {
                (img_ptr->getValues(0))[y * img_w + x] = static_cast<double>(pixels[y][x])
                        / max_pval;
                (img_ptr->getValues(1))[y * img_w + x] = static_cast<double>(pixels[y][x])
                        / max_pval;
                (img_ptr->getValues(2))[y * img_w + x] = static_cast<double>(pixels[y][x])
                        / max_pval;
            }
        }

        ppm_freearray(pixels, img_h);
        fclose(fp);
        return img_ptr;
    }
}
