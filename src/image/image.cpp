#include "image/image.hpp"
#include "utils/make_unique.hpp"
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
				(img_ptr->getValues(0))[(img_width - 1 - h) * img_width + w]
						= raw_data[3*w + 2] / 255.0;
				(img_ptr->getValues(1))[(img_width - 1 - h) * img_width + w]
						= raw_data[3*w + 1] / 255.0;
				(img_ptr->getValues(2))[(img_width - 1 - h) * img_width + w]
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
	
	ImageStruct shrinkImage(ImageStruct& image, int w, int h)
	{
		/* TODO */
	}
	
	ImageStruct cropImage(ImageStruct& image, int x, int y, int w, int h)
	{
		/* TODO */
	}
	
	ImageStruct loadJPEGImage(const char* filepath)
	{
		/* TODO */
	}
}
