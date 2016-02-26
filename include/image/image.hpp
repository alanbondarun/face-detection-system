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
	
	struct ImageStruct
	{
		int width, height, channel_num;
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
	std::unique_ptr<Image> loadBitmapImage(const char* filepath);
	std::unique_ptr<Image> loadBitmapImage(const std::string& filepath);
	
	/* use this functions like: convertToImageStruct(loadBitmapImage("a.bmp")); */
	ImageStruct convertToImageStruct(std::unique_ptr<Image> image_ptr);
	
	/**
	 * Shrinks the given image into the given ratio
	 */
	ImageStruct shrinkImage(ImageStruct& image, double ratio);
	
	/**
	 * Crops the given image.
	 * input: an image struct, (x, y) coordinate of the left-upside corner of the image,
	 *     and width/height of the patch
	 * output: new cropped image
	 */
	ImageStruct cropImage(ImageStruct& image, int x, int y, int w, int h);
	
	/**
	 * Loads a JPEG image into a built-in array of vectors.
	 * input: a string indicating the file path of the image
	 * output: an Image struct including a dynamically allocated double array.
	 *     this contains RGB values for each pixel, each of which are between
	 *     0 and 1 (inclusive).
	 *     the pixels are sorted in raster scan order.
	 */
	ImageStruct loadJPEGImage(const char* filepath);
}

#endif // __LOAD_IMAGE_HPP
