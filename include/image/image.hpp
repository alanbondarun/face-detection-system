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
        size_t getWidth() const { return width; }
        size_t getHeight() const { return height; }
        size_t getChannelNum() const { return channel_num; }
        double* getValues(size_t channel) const { return value[channel]; }
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
    std::unique_ptr<Image> loadBitmapImage(const char* filepath);
    std::unique_ptr<Image> loadBitmapImage(const std::string& filepath);

    /**
     * Shrinks the given image into the given ratio
     */
    std::unique_ptr<Image> shrinkImage(const std::unique_ptr<Image>& image, int w);

    /**
     * Crops the given image.
     * input: an image struct, (x, y) coordinate of the left-upside corner of the image,
     *     and width/height of the patch
     * output: new cropped image
     */
    std::unique_ptr<Image> cropImage(const std::unique_ptr<Image>& image, int x, int y, int w, int h);

    std::unique_ptr<Image> fitImageTo(const std::unique_ptr<Image>& image, int w, int h);

    /**
     * Converts the given image into a grayscale one.
     */
    std::unique_ptr<Image> grayscaleImage(const std::unique_ptr<Image>& image);

    /**
     * Histogram equalization (accepts only grayscale images, though...)
     */
    std::unique_ptr<Image> equalizePatch(const std::unique_ptr<Image>& image);

    /**
     * Image patch preprocessing with least-square fit from [3] (see google docs)
     */
    std::unique_ptr<Image> leastSquarePatch(const std::unique_ptr<Image>& image);

    /**
     * Loads a JPEG image into a built-in array of vectors.
     * input: a string indicating the file path of the image
     * output: an Image struct including a dynamically allocated double array.
     *     this contains RGB values for each pixel, each of which are between
     *     0 and 1 (inclusive).
     *     the pixels are sorted in raster scan order.
     */
    std::unique_ptr<Image> loadJPEGImage(const char* filepath);

    /**
     * Loads a PPM format image file.
     */
    std::unique_ptr<Image> loadPPMImage(const char* filepath);

    /**
     * Loads a PGM format image file.
     */
    std::unique_ptr<Image> loadPGMImage(const char* filepath);

    /**
     * Exports the given image as a PPM file, and returns whether succeeded
     */
    bool saveAsPPM(const std::unique_ptr<Image>& img_ptr, const char* filepath);
}

#endif // __LOAD_IMAGE_HPP
