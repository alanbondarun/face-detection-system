#include "image/image.hpp"
#include "image/image_util.hpp"

namespace NeuralNet
{
    std::vector< std::unique_ptr<Image> > pyramidImage(
            const std::unique_ptr<Image>& original_image,
            double ratio, size_t min_width)
    {
        auto ow = original_image->getWidth();
        auto current_width = ow;
        double bratio = 1.0;
        std::vector< std::unique_ptr<Image> > img_list;

        while (current_width >= min_width)
        {
            img_list.push_back(std::move(shrinkImage(original_image, current_width)));

            bratio *= ratio;
            current_width = static_cast<unsigned int>(current_width * bratio);
        }

        return img_list;
    }

    std::vector< std::unique_ptr<Image> > extractPatches(
            const std::unique_ptr<Image>& original_image,
            size_t patch_width, size_t patch_height, size_t gap)
    {
        auto ow = original_image->getWidth();
        auto oh = original_image->getHeight();
        std::vector< std::unique_ptr<Image> > patch_list;

        for (size_t y = 0; y + patch_height <= oh; y += gap)
        {
            for (size_t x = 0; x + patch_width <= ow; x += gap)
            {
                patch_list.push_back(std::move(cropImage(original_image, x, y,
                        patch_width, patch_height)));
            }
        }

        return patch_list;
    }
}
