#include "image/image.hpp"
#include "image/image_util.hpp"

namespace NeuralNet
{
    std::vector< std::unique_ptr<Image> > pyramidImage(
            const std::unique_ptr<Image>& original_image,
            double ratio, size_t min_width)
    {
        // TODO
    }

    std::vector< std::unique_ptr<Image> > extractPatches(
            const std::unique_ptr<Image>& original_image,
            int patch_width, int patch_height, int stride)
    {
        // TODO
    }
}
