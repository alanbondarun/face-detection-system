#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <cmath>
#include "netpbm/pm.h"
#include "utils/make_unique.hpp"
#include "utils/cl_exception.hpp"
#include "json/json.h"
#include "image/image.hpp"
#include "image/image_util.hpp"
#include "network.hpp"

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/cl.hpp"
#include "cl_context.hpp"

struct SearchConfig
{
    size_t width;
    size_t height;
    size_t patch;
    size_t stride;
    size_t min_image_width;
    float shrink_ratio;
    std::string net_config_file;
    std::string test_file;
    bool grayscale;
    bool uses_gpu;
};

std::unique_ptr<NeuralNet::Image> preprocessImage(
        const std::unique_ptr<NeuralNet::Image>& image)
{
    return NeuralNet::intensityPatch(NeuralNet::leastSquarePatch(
                NeuralNet::equalizePatch(image)));
}

bool loadConfig(SearchConfig& config, const std::string& filepath)
{
    try
    {
        Json::CharReaderBuilder builder;
        builder["collectComments"] = false;

        std::fstream settingStream(filepath, std::ios_base::in);

        Json::Value value;
        std::string errors;
        bool ok = Json::parseFromStream(builder, settingStream, &value, &errors);
        if (!ok)
        {
            std::cout << "error loading " << filepath << ": " <<
                errors << std::endl;
            return false;
        }

        config.width = value["width"].asUInt();
        config.height = value["height"].asUInt();
        config.patch = value["patch_size"].asUInt();
        config.stride = value["stride"].asUInt();
        config.min_image_width = value["min_image_width"].asUInt();
        config.shrink_ratio = value["shrink_ratio"].asDouble();
        config.net_config_file = value["net_config_file"].asString();
        config.test_file = value["test_file"].asString();
        config.grayscale = value["grayscale"].asBool();
        config.uses_gpu = value["uses_gpu"].asBool();
    }
    catch (Json::Exception e)
    {
        std::cout << e.what() << std::endl;
        return false;
    }

    return true;
}

std::unique_ptr<NeuralNet::Network> loadNetwork(const std::string& filepath)
{
    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;

    std::fstream dataStream(filepath, std::ios_base::in);

    Json::Value value;
    std::string error;
    bool ok = Json::parseFromStream(builder, dataStream, &value, &error);
    if (!ok)
    {
        std::cout << "error dealing with network: " << error << std::endl;
        return std::unique_ptr<NeuralNet::Network>();
    }

    return std::make_unique<NeuralNet::Network>(value);
}

std::vector<cl::Image2D> createImagePyramid(const SearchConfig& config,
        cl::Image2D img_buf)
{
    const int orig_w = img_buf.getImageInfo<CL_IMAGE_WIDTH>();
    const int orig_h = img_buf.getImageInfo<CL_IMAGE_HEIGHT>();
    auto context = NeuralNet::CLContext::getInstance().getContext();
    auto queue = NeuralNet::CLContext::getInstance().getCommandQueue();
    auto program = NeuralNet::CLContext::getInstance().getProgram();
    cl_int err;

    cl::ImageFormat img_fmt;
    if (config.grayscale)
    {
        img_fmt = {CL_INTENSITY, CL_FLOAT};
    }
    else
    {
        img_fmt = {CL_RGBA, CL_FLOAT};
    }

    cl::Kernel shrink_image_kernel(program, "shrink_image");
    shrink_image_kernel.setArg(0, img_buf);

    std::vector<cl::Image2D> pyrm_imgs;
    float sratio = 1.0;
    while (orig_w * sratio >= config.min_image_width)
    {
        int shrink_width = orig_w * sratio;
        int shrink_height = (shrink_width * orig_h) / orig_w;

        pyrm_imgs.emplace_back(context, CL_MEM_READ_WRITE, img_fmt, shrink_width, shrink_height);
        shrink_image_kernel.setArg(1, pyrm_imgs.back());

        err = queue.enqueueNDRangeKernel(shrink_image_kernel, cl::NullRange,
                cl::NDRange(shrink_width, shrink_height), cl::NullRange);
        NeuralNet::printError(err, "enqueueNDRangeKernel");

        sratio *= config.shrink_ratio;
    }

    return pyrm_imgs;
}

void clGetPatches(const SearchConfig& config, const std::unique_ptr<NeuralNet::Image>& input_img,
        std::vector<float>& patch_data)
{
    const cl::ImageFormat img_fmt{CL_RGBA, CL_FLOAT};
    const cl::ImageFormat gray_img_fmt{CL_INTENSITY, CL_FLOAT};
    auto context = NeuralNet::CLContext::getInstance().getContext();
    auto queue = NeuralNet::CLContext::getInstance().getCommandQueue();
    auto program = NeuralNet::CLContext::getInstance().getProgram();
    cl_int err;

    auto img_w = input_img->getWidth();
    auto img_h = input_img->getHeight();

    auto img_mixed_vals = input_img->getPaddedMixedValues(4, 1.0);
    cl::Image2D in_img_buf(context, CL_MEM_READ_ONLY, img_fmt,
            img_w, img_h, 0, nullptr, &err);
    NeuralNet::printError(err, "in_img_buf");

    cl::size_t<3> in_offset, in_region;
    in_offset[0] = in_offset[1] = in_offset[2] = 0;
    in_region[0] = input_img->getWidth();
    in_region[1] = input_img->getHeight();
    in_region[2] = 1;

    err = queue.enqueueWriteImage(in_img_buf, CL_TRUE, in_offset, in_region,
            img_w * 4 * sizeof(float), 0,
            img_mixed_vals.data());
    NeuralNet::printError(err, "enqueueReadImage");

    // phase 1) grayscaling the input image
    if (config.grayscale)
    {
        cl::Image2D gray_img_buf(context, CL_MEM_READ_WRITE, gray_img_fmt,
                img_w, img_h, 0, nullptr, &err);
        NeuralNet::printError(err, "gray_img_buf");

        cl::Kernel grayscale_image_kernel(program, "grayscale_img");
        grayscale_image_kernel.setArg(0, in_img_buf);
        grayscale_image_kernel.setArg(1, gray_img_buf);

        err = queue.enqueueNDRangeKernel(grayscale_image_kernel, cl::NullRange,
                cl::NDRange(img_w, img_h), cl::NullRange);
        NeuralNet::printError(err, "grayscale_img kernel execution");

        in_img_buf = gray_img_buf;
    }

    // phase 2) creating image pyramid
    auto pyrm_imgs = createImagePyramid(config, in_img_buf);

    // phase 3) create image patches
    cl::Kernel extract_patch_kernel(program, "extract_image_patches");
    for (auto& pyrm_img_buf: pyrm_imgs)
    {
        int img_w = pyrm_img_buf.getImageInfo<CL_IMAGE_WIDTH>();
        int img_h = pyrm_img_buf.getImageInfo<CL_IMAGE_HEIGHT>();
        int i_patch = static_cast<int>(config.patch);
        int i_stride = static_cast<int>(config.stride);
        int patches_w = (img_w - config.patch) / config.stride + 1;
        int patches_h = (img_h - config.patch) / config.stride + 1;
        int patch_nums = i_patch * i_patch;

        int real_channels = 3;
        if (config.grayscale)
            real_channels = 1;

        std::vector<cl::Buffer> patch_bufs;
        for (int j=0; j<patches_h; j++)
        {
            for (int i=0; i<patches_w; i++)
            {
                patch_bufs.emplace_back(context, CL_MEM_READ_WRITE,
                        sizeof(float) * config.patch * config.patch * real_channels);

                extract_patch_kernel.setArg(0, pyrm_img_buf);
                extract_patch_kernel.setArg(1, patch_bufs.back());
                extract_patch_kernel.setArg(2, sizeof(int), &i_patch);
                extract_patch_kernel.setArg(3, sizeof(int), &i_patch);
                extract_patch_kernel.setArg(4, sizeof(int), &i);
                extract_patch_kernel.setArg(5, sizeof(int), &j);
                extract_patch_kernel.setArg(6, sizeof(int), &i_stride);

                err = queue.enqueueNDRangeKernel(extract_patch_kernel,
                        cl::NullRange,
                        cl::NDRange(config.patch * config.patch *
                            real_channels),
                        cl::NullRange);
                NeuralNet::printError(err, "enqueNDRangeKernel");
            }
        }

        for (auto& patch_buf: patch_bufs)
        {
            auto new_img = std::make_unique<NeuralNet::Image>(
                    config.patch, config.patch, real_channels);

            for (size_t c=0; c<new_img->getChannelNum(); c++)
            {
                err = queue.enqueueReadBuffer(patch_buf, CL_TRUE,
                        sizeof(float) * patch_nums * c,
                        sizeof(float) * patch_nums,
                        new_img->getValues(c));
                NeuralNet::printError(err, "enqueueReadBuffer");
            }

            if (config.grayscale)
            {
                // TODO: preprocessing with OpenCL?
                auto pre_img = preprocessImage(new_img);
                patch_data.insert(patch_data.end(),
                        pre_img->getValues(0),
                        pre_img->getValues(0) + (config.patch * config.patch));
            }
            else
            {
                for (size_t c=0; c<new_img->getChannelNum(); c++)
                {
                    patch_data.insert(patch_data.end(),
                            new_img->getValues(c),
                            new_img->getValues(c) + (config.patch * config.patch));
                }
            }
        }
    }
}

void cpuGetPatches(const SearchConfig& config,
        const std::unique_ptr<NeuralNet::Image>& input_img,
        std::vector<float>& patch_data)
{
    auto image_pyramid = NeuralNet::pyramidImage(input_img, config.shrink_ratio,
            config.min_image_width);
    std::cout << "(cpu) loading image pyramid complete" << std::endl;

    for (auto& small_image: image_pyramid)
    {
        auto patch_list = NeuralNet::extractPatches(small_image, config.patch,
                config.patch, config.stride);
        if (config.grayscale)
        {
            for (auto& patch_ptr: patch_list)
            {
                auto gray_patch_ptr = preprocessImage(
                        NeuralNet::grayscaleImage(patch_ptr));
                auto gray_data = gray_patch_ptr->getValues(0);

                for (size_t i = 0; i < gray_patch_ptr->getWidth() *
                        gray_patch_ptr->getHeight(); i++)
                {
                    patch_data.push_back(gray_data[i]);
                }
            }
        }
        else
        {
            for (auto& patch_ptr: patch_list)
            {
                for (size_t c = 0; c < patch_ptr->getChannelNum(); c++)
                {
                    for (size_t i = 0; i < patch_ptr->getWidth() *
                            patch_ptr->getHeight(); i++)
                    {
                        patch_data.push_back((patch_ptr->getValues(c))[i]);
                    }
                }
            }
        }
    }
}

int accumulatePatches(const SearchConfig& config,
        const std::vector<int>& category_list)
{
    std::vector<std::pair<size_t, size_t>> pyramid_patches;
    float delta = 1.0;
    while (config.width * delta >= config.min_image_width)
    {
        size_t tw = config.width * delta;
        size_t th = config.height * delta;
        size_t nw = (tw - config.patch) / config.stride + 1;
        size_t nh = (th - config.patch) / config.stride + 1;
        pyramid_patches.emplace_back(nw, nh);

        delta *= config.shrink_ratio;
    }

    // phase 1) construct the delta map
    std::vector<int> deltamap;
    deltamap.resize((config.width + 1) * (config.height + 1), 0);
    size_t map_w = config.width + 1;
    size_t map_h = config.height + 1;
    for (size_t i = 0; i < category_list.size(); i++)
    {
        if (category_list[i] != 0)
            continue;

        // find the level index and (x,y) coordinate of the patch
        size_t j = 0;
        size_t sump = 0;
        while (j < pyramid_patches.size())
        {
            if (sump + pyramid_patches[j].first * pyramid_patches[j].second > i)
                break;
            sump += pyramid_patches[j].first * pyramid_patches[j].second;
            j++;
        }

        float sratio = std::pow(config.shrink_ratio, j);
        float patch_size = config.patch * sratio;
        float stride_size = config.stride * sratio;
        float posx = ((i - sump) % (pyramid_patches[j].second)) * stride_size;
        float posy = ((i - sump) / (pyramid_patches[j].second)) * stride_size;

        size_t iposx = static_cast<size_t>(posx);
        size_t iposy = static_cast<size_t>(posy);
        size_t posa = std::min(static_cast<size_t>(posx + patch_size),
                config.width);
        size_t posb = std::min(static_cast<size_t>(posy + patch_size),
                config.height);

        deltamap[iposy * map_w + iposx]++;
        deltamap[iposy * map_w + posa]--;
        deltamap[posb * map_w + iposx]--;
        deltamap[posb * map_w + posa]++;
    }

    // phase 2) accumulate the delta map
    for (size_t i = 1; i < map_w; i++)
    {
        deltamap[i] += deltamap[i-1];
    }
    for (size_t j = 1; j < map_h; j++)
    {
        deltamap[j*map_w] += deltamap[(j-1)*map_w];
        for (size_t i = 1; i < map_w; i++)
        {
            deltamap[j*map_w + i] += (deltamap[(j-1)*map_w + i]
                    + deltamap[j*map_w + (i-1)]
                    - deltamap[(j-1)*map_w + (i-1)]);
        }
    }

    // phase 3) find the max value in the accumulated map
    int mval = 0;
    for (auto& val: deltamap)
    {
        mval = std::max(mval, val);
    }
    return mval;
}

int main(int argc, char* argv[])
{
    pm_init(argv[0], 0);

    SearchConfig config;
    if (!loadConfig(config, "../net_set/search_face.json"))
    {
        std::cout << "error at loading configuration" << std::endl;
        return 0;
    }
    std::cout << "loading configuration complete" << std::endl;

    // load an image (just an empty pointer now)
    std::chrono::time_point<std::chrono::system_clock> time_point_start, time_point_finish;
    std::chrono::duration<double> elapsed_seconds;
    time_point_start = std::chrono::system_clock::now();

    auto image = NeuralNet::loadBitmapImage(config.test_file.c_str());

    time_point_finish = std::chrono::system_clock::now();
    elapsed_seconds = time_point_finish - time_point_start;
    std::cout << "loading test image complete, time: " << elapsed_seconds.count()
        << "s" << std::endl;
    
    // load image patches from the given image
    time_point_start = std::chrono::system_clock::now();
    std::vector<float> patch_data;
    if (config.uses_gpu)
    {
        clGetPatches(config, image, patch_data);
    }
    else
    {
        cpuGetPatches(config, image, patch_data);
    }
    time_point_finish = std::chrono::system_clock::now();
    elapsed_seconds = time_point_finish - time_point_start;
    std::cout << "loading image patches complete, time: " << elapsed_seconds.count()
        << "s" << std::endl;
    
    // classify patches with the trained network
    auto network = loadNetwork(config.net_config_file);
    if (!network)
    {
        std::cout << "error at loadNetwork()" << std::endl;
        return 0;
    }
    network->loadFromFiles();
    std::cout << "loading network complete" << std::endl;
    
    time_point_start = std::chrono::system_clock::now();

    auto category_list = network->evaluateAll(patch_data);

    time_point_finish = std::chrono::system_clock::now();
    elapsed_seconds = time_point_finish - time_point_start;
    std::cout << "image patch evaluation complete, time: " << elapsed_seconds.count()
        << "s" << std::endl;

    size_t positive_res = 0;
    for (auto val: category_list[0])
    {
        if (val == 0)
            positive_res++;
    }

    std::cout << "positive patches = " << positive_res << std::endl;

    time_point_start = std::chrono::system_clock::now();
    auto score = accumulatePatches(config, category_list[0]);
    time_point_finish = std::chrono::system_clock::now();
    elapsed_seconds = time_point_finish - time_point_start;
    std::cout << "score = " << score << ", score calc time: "
        << elapsed_seconds.count() << std::endl;
}
