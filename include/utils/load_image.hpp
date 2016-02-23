#ifndef __LOAD_IMAGE_HPP
#define __LOAD_IMAGE_HPP

#include <cstdlib>
#include <string>
#include <utility>
#include <memory>

namespace NeuralNet
{
	/**
	 * a struct descripting an image.
	 */
	class Image
	{
	public:
		explicit Image(size_t _width, size_t _height, size_t _channels);
		~Image();
		size_t getWidth() { return width; }
		size_t getHeight() { return height; }
		size_t getChannelNum() { return channel_num; }
		double* getValues(size_t channel) { return value[channel]; }
	private:
		size_t width, height, channel_num;
		double** value;
	};

	/**
	 * Loads a image into a built-in array of vectors.
	 * input: a string indicating the file path of the image
	 * output: an Image struct including a dynamically allocated double array.
	 *     this contains RGB values for each pixel, each of which are between
	 *     0 and 1 (inclusive).
	 *     the pixels are sorted in raster scan order.
	 */
	std::unique_ptr<Image> loadImage(const char* filepath);
	std::unique_ptr<Image> loadImage(const std::string& filepath);
}

#endif // __LOAD_IMAGE_HPP
