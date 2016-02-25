#include "image/image.hpp"
#include <iostream>
#include <utility>

int main()
{
	auto img_ptr = std::move(NeuralNet::loadBitmapImage("test.bmp"));
	
	if (!img_ptr)
	{
		std::cout << "error loading image" << std::endl;
		return 0;
	}
	
	size_t ww = img_ptr->getWidth(), hh = img_ptr->getHeight();
	std::cout << "w=" << ww << ", h=" << hh << std::endl;

	for (size_t h = 0; h < hh; h++)
	{
		for (size_t w = 0; w < ww; w++)
		{
			std::cout << "(" << (img_ptr->getValues(0))[h * ww + w]
					<< "," << (img_ptr->getValues(1))[h * ww + w]
					<< "," << (img_ptr->getValues(2))[h * ww + w] << ") ";
		}
		std::cout << std::endl;
	}
	
	return 0;
}
