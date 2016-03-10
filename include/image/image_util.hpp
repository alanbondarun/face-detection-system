#ifndef __IMAGE_UTIL_HPP
#define __IMAGE_UTIL_HPP

#include <vector>
#include <memory>
#include "image/image.hpp"

namespace NeuralNet
{
    /**
     * Make a set of gradually-decreasing images
     */
    std::vector< std::unique_ptr<Image> > pyramidImage(
            const std::unique_ptr<Image>& original_image,
            double ratio, size_t min_width);

    /**
     * Raster-scan a given image to extract a set of image patches
     */
    std::vector< std::unique_ptr<Image> > extractPatches(
            const std::unique_ptr<Image>& original_image,
            int patch_width, int patch_height, int stride);
}

#endif // __IMAGE_UTIL_HPP
