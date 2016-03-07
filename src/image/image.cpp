#include "image/image.hpp"
#include "utils/make_unique.hpp"
#include "netpbm/ppm.h"
#include "netpbm/pgm.h"
#include "Eigen/Dense"
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

        for (size_t ch = 0; ch < image->getChannelNum(); ch++)
        {
            auto orig_data_ptr = image->getValues(ch);
            auto res_data_ptr = img_ptr->getValues(ch);

            for (int j = 0; j < h; j++)
            {
                for (int i = 0; i < w; i++)
                {
                    double posx = static_cast<double>(i * orig_w) / w;
                    double posy = static_cast<double>(j * orig_h) / h;

                    int iposx = static_cast<int>(posx);
                    int iposy = static_cast<int>(posy);

                    double dposx = posx - static_cast<double>(iposx);
                    double dposy = posy - static_cast<double>(iposy);

                    double val = 0;
                    if (iposx + 1 >= orig_w)
                    {
                        if (iposy + 1 >= orig_h)
                        {
                            val += orig_data_ptr[iposy * orig_w + iposx];
                        }
                        else
                        {
                            val += (1-dposy) * orig_data_ptr[iposy * orig_w + iposx];
                            val += (dposy) * orig_data_ptr[(iposy+1) * orig_w + iposx];
                        }
                    }
                    else if (iposy + 1 >= orig_h)
                    {
                        val += (1-dposx) * (orig_data_ptr[iposy * orig_w + iposx]);
                        val += (dposx) * (orig_data_ptr[iposy * orig_w + iposx + 1]);
                    }
                    else
                    {
                        val += (1-dposx) * (1-dposy) * (orig_data_ptr[iposy * orig_w + iposx]);
                        val += (dposx) * (1-dposy) * (orig_data_ptr[iposy * orig_w + iposx + 1]);
                        val += (1-dposx) * (dposy) * (orig_data_ptr[(iposy+1) * orig_w + iposx]);
                        val += (dposx) * (dposy) * (orig_data_ptr[(iposy+1) * orig_w + iposx + 1]);
                    }
                    res_data_ptr[j*w + i] = val;
                }
            }
        }

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
        int tw = image->getWidth();
        int th = image->getHeight();
        int pw = (w * image->getHeight()) / h;
        int ph = (h * image->getWidth()) / w;

        if (ph <= th)
        {
            auto middle_img = cropImage(image, 0, th/2 - ph/2, tw, ph);
            return shrinkImage(middle_img, w);
        }
        auto middle_img = cropImage(image, tw/2 - pw/2, 0, pw, th);
        return shrinkImage(middle_img, w);
    }

    std::unique_ptr<Image> grayscaleImage(const std::unique_ptr<Image>& image)
    {
        const auto w = image->getWidth();
        const auto h = image->getHeight();
        const auto nch = image->getChannelNum();

        if (nch < 3)
            return std::unique_ptr<Image>();

        auto res_img = std::make_unique<Image>(w, h, 1);
        auto valptr = res_img->getValues(0);

        double ch_coeff[] = {0.2125, 0.7154, 0.0721};
        for (size_t y = 0; y < h; y++)
        {
            for (size_t x = 0; x < w; x++)
            {
                valptr[y*w + x] = 0;
                for (size_t c = 0; c < 3; c++)
                {
                    valptr[y*w + x] += ch_coeff[c] * (image->getValues(c)[y*w + x]);
                }
            }
        }
        return res_img;
    }

    std::unique_ptr<Image> equalizePatch(const std::unique_ptr<Image>& image)
    {
        if (image->getChannelNum() != 1)
            return std::unique_ptr<Image>();

        const size_t level = 255;
        const auto w = image->getWidth();
        const auto h = image->getHeight();

        std::vector<size_t> val_distrib(level + 1, 0);
        for (size_t j = 0; j < h; j++)
        {
            for (size_t i = 0; i < w; i++)
            {
                int idx = static_cast<int>((image->getValues(0))[j*w + i] * level);
                if (idx < 0)
                    val_distrib[0]++;
                else if (idx > level)
                    val_distrib[level]++;
                else
                    val_distrib[idx]++;
            }
        }

        std::vector<size_t> val_cdf(level + 1, 0);
        val_cdf[0] = val_distrib[0];
        for (size_t i = 1; i <= level; i++)
        {
            val_cdf[i] = val_cdf[i-1] + val_distrib[i];
        }

        auto newImage = std::make_unique<Image>(w, h, 1);
        auto new_data_ptr = newImage->getValues(0);
        for (size_t j = 0; j < h; j++)
        {
            for (size_t i = 0; i < w; i++)
            {
                int idx = static_cast<int>((image->getValues(0))[j*w + i] * level);
                if (idx < 0)
                    new_data_ptr[j*w + i] = static_cast<double>(val_cdf[0]) / level;
                else if (idx > level)
                    new_data_ptr[j*w + i] = static_cast<double>(val_cdf[level]) / level;
                else
                    new_data_ptr[j*w + i] = static_cast<double>(val_cdf[idx]) / level;
            }
        }
        return newImage;
    }

    std::unique_ptr<Image> leastSquarePatch(const std::unique_ptr<Image>& image)
    {
        // process grayscale patches only
        if (image->getChannelNum() > 1)
            return std::unique_ptr<Image>();

        auto data_ptr = image->getValues(0);
        const auto w = image->getWidth();
        const auto h = image->getHeight();

        // least square fit to linear plane for light compensation
        double xsum_orig = 0;
        double ysum_orig = 0;
        double csum_orig = 0;
        for (size_t j = 0; j < h; j++)
        {
            for (size_t i = 0; i < w; i++)
            {
                xsum_orig += (i * data_ptr[j*w + i]);
                ysum_orig += (j * data_ptr[j*w + i]);
                csum_orig += (data_ptr[j*w + i]);
            }
        }
        Eigen::Vector3d vsum(xsum_orig, ysum_orig, csum_orig);

        double x2sum = 0, y2sum = 0, xysum = 0, xsum = 0, ysum = 0;
        double csum = w*h;
        for (size_t j = 0; j < h; j++)
        {
            for (size_t i = 0; i < w; i++)
            {
                x2sum += (i*i);
                y2sum += (j*j);
                xysum += (i*j);
                xsum += i;
                ysum += j;
            }
        }
        Eigen::Matrix3d msum;
        msum << x2sum, xysum, xsum,
             xysum, y2sum, ysum,
             xsum, ysum, csum;
        
        auto vcoeff = msum.inverse() * vsum;
        
        auto newImage = std::make_unique<Image>(w, h, 1);
        for (size_t j = 0; j < h; j++)
        {
            for (size_t i = 0; i < w; i++)
            {
                (newImage->getValues(0))[j*w + i] = data_ptr[j*w + i]
                    - i*vcoeff[0] - j*vcoeff[1] - vcoeff[2];
            }
        }
        return newImage;
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

    bool saveAsPPM(const std::unique_ptr<Image>& img_ptr, const char* filepath)
    {
        FILE* fp = fopen(filepath, "wb");
        if (!fp)
        {
            printf("error loading file %s at saveAsPPM()\n", filepath);
            return false;
        }

        const size_t w = img_ptr->getWidth();
        const size_t h = img_ptr->getHeight();
        const pixval maxVal = 255;

        pixel** pxls = ppm_allocarray(w, h);
        if (!pxls)
        {
            printf("error allocating memory space at saveAsPPM()");
            fclose(fp);
            return false;
        }

        for (size_t y = 0; y < h; y++)
        {
            for (size_t x = 0; x < w; x++)
            {
                if (img_ptr->getChannelNum() >= 3)
                {
                    double vclip[3];
                    for (size_t c = 0; c < 3; c++)
                    {
                        vclip[c] = (img_ptr->getValues(c))[y*w + x];
                        if (vclip[c] < 0)
                            vclip[c] = 0;
                        if (vclip[c] > 1)
                            vclip[c] = 1;
                    }
                    pxls[y][x].r = static_cast<pixval>(vclip[0] * maxVal);
                    pxls[y][x].g = static_cast<pixval>(vclip[1] * maxVal);
                    pxls[y][x].b = static_cast<pixval>(vclip[2] * maxVal);
                }
                else
                {
                    double vclip = (img_ptr->getValues(0))[y*w + x];
                    if (vclip < 0)
                        vclip = 0;
                    if (vclip > 1)
                        vclip = 1;
                    pxls[y][x].r = static_cast<pixval>(vclip * maxVal);
                    pxls[y][x].g = static_cast<pixval>(vclip * maxVal);
                    pxls[y][x].b = static_cast<pixval>(vclip * maxVal);
                }
            }
        }

        ppm_writeppm(fp, pxls, w, h, maxVal, 0);
        ppm_freearray(pxls, h);
        fclose(fp);
        return true;
    }
}
