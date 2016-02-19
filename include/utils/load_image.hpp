#ifndef __LOAD_IMAGE_HPP
#define __LOAD_IMAGE_HPP

#include <cstdlib>

namespace NeuralNet
{
	/**
	 * a struct descripting an image.
	 */
	struct Image
	{
		size_t width, height;
		double* value;
	};
	
	/**
	 * Loads a image into a built-in array of vectors.
	 * input: a string indicating the file path of the image
	 * output: an Image struct including a dynamically allocated double array.
	 *     this contains RGB values for each pixel, each of which are between
	 *     0 and 1 (inclusive).
	 *     the pixels are sorted in raster scan order.
	 */
	Image loadImage(const char* filepath);
}

#endif // __LOAD_IMAGE_HPP