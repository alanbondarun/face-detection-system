#include "face_finder.hpp"
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

namespace NeuralNet
{
    FaceFinder::FaceFinder(const std::string& m_config_path)
    {
        if (!loadConfig(m_config_path))
            throw FaceFinderException("loading m_configuration");
        std::cout << "loading m_configuration complete" << std::endl;

        // classify patches with the trained network
        loadNetwork(m_config.net_config_file);
        if (!m_network)
            throw FaceFinderException("loading Network");
        m_network->loadFromFiles();
        std::cout << "loading network complete" << std::endl;
    }

    std::unique_ptr<Image> FaceFinder::preprocessImage(
            const std::unique_ptr<Image>& image)
    {
        return intensityPatch(leastSquarePatch(equalizePatch(image)));
    }

    bool FaceFinder::loadConfig(const std::string& filepath)
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

            m_config.width = value["width"].asUInt();
            m_config.height = value["height"].asUInt();
            m_config.patch = value["patch_size"].asUInt();
            m_config.stride = value["stride"].asUInt();
            m_config.min_image_width = value["min_image_width"].asUInt();
            m_config.shrink_ratio = value["shrink_ratio"].asDouble();
            m_config.net_config_file = value["net_config_file"].asString();
            m_config.grayscale = value["grayscale"].asBool();
            m_config.uses_gpu = value["uses_gpu"].asBool();
        }
        catch (Json::Exception e)
        {
            std::cout << e.what() << std::endl;
            return false;
        }

        return true;
    }

    void FaceFinder::loadNetwork(const std::string& filepath)
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
        }

        m_network = std::make_unique<Network>(value);
    }

    std::vector<cl::Image2D> FaceFinder::createImagePyramid(
            cl::Image2D& img_buf)
    {
        const int orig_w = img_buf.getImageInfo<CL_IMAGE_WIDTH>();
        const int orig_h = img_buf.getImageInfo<CL_IMAGE_HEIGHT>();
        auto context = CLContext::getInstance().getContext();
        auto queue = CLContext::getInstance().getCommandQueue();
        auto program = CLContext::getInstance().getProgram();
        cl_int err;

        cl::ImageFormat img_fmt;
        if (m_config.grayscale)
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
        while (orig_w * sratio >= m_config.min_image_width)
        {
            int shrink_width = orig_w * sratio;
            int shrink_height = (shrink_width * orig_h) / orig_w;

            pyrm_imgs.emplace_back(context, CL_MEM_READ_WRITE, img_fmt,
                    shrink_width, shrink_height);
            shrink_image_kernel.setArg(1, pyrm_imgs.back());

            err = queue.enqueueNDRangeKernel(shrink_image_kernel, cl::NullRange,
                    cl::NDRange(shrink_width, shrink_height), cl::NullRange);
            printError(err, "enqueueNDRangeKernel");

            sratio *= m_config.shrink_ratio;
        }

        return pyrm_imgs;
    }

    cl::Image3D FaceFinder::preprocessGrayscalePatches(cl::Image3D& patch_buf)
    {
        const int level = 256;
        const float norm_mean = 0;
        const float norm_stdev = 0.25;

        const cl::ImageFormat gray_img_fmt{CL_INTENSITY, CL_FLOAT};
        auto context = CLContext::getInstance().getContext();
        auto queue = CLContext::getInstance().getCommandQueue();
        auto program = CLContext::getInstance().getProgram();
        cl_int err;

        int npatch_w = patch_buf.getImageInfo<CL_IMAGE_HEIGHT>();
        int npatch_h = patch_buf.getImageInfo<CL_IMAGE_DEPTH>();
        int ipatch = static_cast<int>(m_config.patch);

        // phase 1) histogram equalization
        cl::Image3D cdf_buf(context, CL_MEM_READ_WRITE, gray_img_fmt,
                level, npatch_w, npatch_h);

        cl::Kernel distrib_kernel(program, "patch_create_cdf");
        distrib_kernel.setArg(0, patch_buf);
        distrib_kernel.setArg(1, cdf_buf);
        err = queue.enqueueNDRangeKernel(distrib_kernel, cl::NullRange,
                cl::NDRange(npatch_w, npatch_h),
                cl::NullRange);
        printError(err, "enqueNDRangeKernel");

        cl::Image3D l1_buf(context, CL_MEM_READ_WRITE, gray_img_fmt,
                m_config.patch * m_config.patch, npatch_w, npatch_h);

        cl::Kernel apply_cdf_kernel(program, "patch_apply_cdf");
        apply_cdf_kernel.setArg(0, patch_buf);
        apply_cdf_kernel.setArg(1, cdf_buf);
        apply_cdf_kernel.setArg(2, l1_buf);
        err = queue.enqueueNDRangeKernel(apply_cdf_kernel, cl::NullRange,
                cl::NDRange(m_config.patch * m_config.patch, npatch_w,
                    npatch_h),
                cl::NullRange);
        printError(err, "enqueNDRangeKernel");

        // phase 2) least-square planar fit
        cl::Image3D lsq_coeff_buf(context, CL_MEM_READ_WRITE, gray_img_fmt,
                3, npatch_w, npatch_h);

        cl::Kernel lsq_coeff_kernel(program, "patch_get_lsq_coeff");
        lsq_coeff_kernel.setArg(0, l1_buf);
        lsq_coeff_kernel.setArg(1, lsq_coeff_buf);
        lsq_coeff_kernel.setArg(2, sizeof(int), &ipatch);
        err = queue.enqueueNDRangeKernel(lsq_coeff_kernel, cl::NullRange,
                cl::NDRange(npatch_w, npatch_h),
                cl::NullRange);
        printError(err, "enqueNDRangeKernel");

        cl::Image3D l2_buf(context, CL_MEM_READ_WRITE, gray_img_fmt,
                m_config.patch * m_config.patch, npatch_w, npatch_h);

        cl::Kernel apply_lsq_kernel(program, "patch_apply_lsq");
        apply_lsq_kernel.setArg(0, l1_buf);
        apply_lsq_kernel.setArg(1, lsq_coeff_buf);
        apply_lsq_kernel.setArg(2, l2_buf);
        apply_lsq_kernel.setArg(3, sizeof(int), &ipatch);
        err = queue.enqueueNDRangeKernel(apply_lsq_kernel, cl::NullRange,
                cl::NDRange(m_config.patch * m_config.patch, npatch_w,
                    npatch_h),
                cl::NullRange);
        printError(err, "enqueNDRangeKernel");

        // phase 3) normalization
        cl::Image3D mv_buf(context, CL_MEM_READ_WRITE, gray_img_fmt,
                2, npatch_w, npatch_h);

        cl::Kernel mv_kernel(program, "patch_get_mean_var");
        mv_kernel.setArg(0, l2_buf);
        mv_kernel.setArg(1, mv_buf);
        err = queue.enqueueNDRangeKernel(mv_kernel, cl::NullRange,
                cl::NDRange(npatch_w, npatch_h),
                cl::NullRange);
        printError(err, "enqueNDRangeKernel");

        cl::Image3D out_buf(context, CL_MEM_READ_WRITE, gray_img_fmt,
                m_config.patch * m_config.patch, npatch_w, npatch_h);

        cl::Kernel norm_kernel(program, "patch_normalize");
        norm_kernel.setArg(0, l2_buf);
        norm_kernel.setArg(1, mv_buf);
        norm_kernel.setArg(2, out_buf);
        norm_kernel.setArg(3, sizeof(int), &ipatch);
        norm_kernel.setArg(4, sizeof(float), &norm_mean);
        norm_kernel.setArg(5, sizeof(float), &norm_stdev);
        err = queue.enqueueNDRangeKernel(norm_kernel, cl::NullRange,
                cl::NDRange(m_config.patch * m_config.patch, npatch_w,
                    npatch_h),
                cl::NullRange);
        printError(err, "enqueNDRangeKernel");

        return out_buf;
    }

    void FaceFinder::clGetPatches(const std::unique_ptr<Image>& input_img,
            std::vector<float>& patch_data)
    {
        const cl::ImageFormat img_fmt{CL_RGBA, CL_FLOAT};
        const cl::ImageFormat gray_img_fmt{CL_INTENSITY, CL_FLOAT};
        const cl::ImageFormat res_img_fmt = (m_config.grayscale)
            ? (gray_img_fmt) : (img_fmt);
        auto context = CLContext::getInstance().getContext();
        auto queue = CLContext::getInstance().getCommandQueue();
        auto program = CLContext::getInstance().getProgram();
        cl_int err;

        auto img_w = input_img->getWidth();
        auto img_h = input_img->getHeight();

        auto img_mixed_vals = input_img->getPaddedMixedValues(4, 1.0);
        cl::Image2D in_img_buf(context, CL_MEM_READ_ONLY, img_fmt,
                img_w, img_h, 0, nullptr, &err);
        printError(err, "in_img_buf");

        cl::size_t<3> in_offset, in_region;
        in_offset[0] = in_offset[1] = in_offset[2] = 0;
        in_region[0] = input_img->getWidth();
        in_region[1] = input_img->getHeight();
        in_region[2] = 1;

        err = queue.enqueueWriteImage(in_img_buf, CL_TRUE, in_offset, in_region,
                img_w * 4 * sizeof(float), 0,
                img_mixed_vals.data());
        printError(err, "enqueueReadImage");

        // phase 1) grayscaling the input image
        if (m_config.grayscale)
        {
            cl::Image2D gray_img_buf(context, CL_MEM_READ_WRITE, gray_img_fmt,
                    img_w, img_h, 0, nullptr, &err);
            printError(err, "gray_img_buf");

            cl::Kernel grayscale_image_kernel(program, "grayscale_img");
            grayscale_image_kernel.setArg(0, in_img_buf);
            grayscale_image_kernel.setArg(1, gray_img_buf);

            err = queue.enqueueNDRangeKernel(grayscale_image_kernel, cl::NullRange,
                    cl::NDRange(img_w, img_h), cl::NullRange);
            printError(err, "grayscale_img kernel execution");

            in_img_buf = gray_img_buf;
        }

        // phase 2) creating image pyramid
        auto pyrm_imgs = createImagePyramid(in_img_buf);

        // phase 3) create image patches
        cl::Kernel extract_patch_kernel(program, "extract_image_patches");
        for (auto& pyrm_img_buf: pyrm_imgs)
        {
            int img_w = pyrm_img_buf.getImageInfo<CL_IMAGE_WIDTH>();
            int img_h = pyrm_img_buf.getImageInfo<CL_IMAGE_HEIGHT>();
            int i_patch = static_cast<int>(m_config.patch);
            int i_stride = static_cast<int>(m_config.stride);
            int patches_w = (img_w - m_config.patch) / m_config.stride + 1;
            int patches_h = (img_h - m_config.patch) / m_config.stride + 1;

            cl::Image3D patch_img_buf(context, CL_MEM_READ_WRITE, res_img_fmt,
                    i_patch * i_patch, patches_w, patches_h);

            extract_patch_kernel.setArg(0, pyrm_img_buf);
            extract_patch_kernel.setArg(1, patch_img_buf);
            extract_patch_kernel.setArg(2, sizeof(int), &i_patch);
            extract_patch_kernel.setArg(3, sizeof(int), &i_stride);

            err = queue.enqueueNDRangeKernel(extract_patch_kernel,
                    cl::NullRange,
                    cl::NDRange(m_config.patch * m_config.patch,
                        patches_w, patches_h),
                    cl::NullRange);
            printError(err, "enqueNDRangeKernel");

            cl::size_t<3> patch_region;
            patch_region[0] = i_patch * i_patch;
            patch_region[1] = patches_w;
            patch_region[2] = patches_h;

            if (m_config.grayscale)
            {
                auto prep_buf = preprocessGrayscalePatches(patch_img_buf);

                std::vector<float> patch_vals(i_patch * i_patch * patches_w *
                        patches_h);
                err = queue.enqueueReadImage(prep_buf, CL_TRUE,
                        in_offset, patch_region,
                        0, 0, patch_vals.data());
                printError(err, "enqueueReadImage");

                patch_data.insert(patch_data.end(),
                        patch_vals.begin(), patch_vals.end());
            }
            else
            {
                std::vector<float> patch_vals(i_patch * i_patch * patches_w *
                        patches_h * 4);
                err = queue.enqueueReadImage(patch_img_buf, CL_TRUE,
                        in_offset, patch_region,
                        0, 0, patch_vals.data());
                printError(err, "enqueueReadImage");

                auto resolved_vals = resolveMixedValues(patch_vals,
                        4, 3, i_patch * i_patch);
                patch_data.insert(patch_data.end(),
                        resolved_vals.begin(), resolved_vals.end());
            }
        }
    }

    void FaceFinder::cpuGetPatches(const std::unique_ptr<Image>& input_img,
            std::vector<float>& patch_data)
    {
        auto image_pyramid = pyramidImage(input_img, m_config.shrink_ratio,
                m_config.min_image_width);

        for (auto& small_image: image_pyramid)
        {
            auto patch_list = extractPatches(small_image, m_config.patch,
                    m_config.patch, m_config.stride);
            if (m_config.grayscale)
            {
                for (auto& patch_ptr: patch_list)
                {
                    auto gray_patch_ptr = preprocessImage(
                            grayscaleImage(patch_ptr));
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

    int FaceFinder::accumulatePatches(const std::vector<int>& category_list)
    {
        std::vector<std::pair<size_t, size_t>> pyramid_patches;
        float delta = 1.0;
        while (m_config.width * delta >= m_config.min_image_width)
        {
            size_t tw = m_config.width * delta;
            size_t th = m_config.height * delta;
            size_t nw = (tw - m_config.patch) / m_config.stride + 1;
            size_t nh = (th - m_config.patch) / m_config.stride + 1;
            pyramid_patches.emplace_back(nw, nh);

            delta *= m_config.shrink_ratio;
        }

        // phase 1) construct the delta map
        std::vector<int> deltamap;
        deltamap.resize((m_config.width + 1) * (m_config.height + 1), 0);
        size_t map_w = m_config.width + 1;
        size_t map_h = m_config.height + 1;
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

            float sratio = std::pow(m_config.shrink_ratio, j);
            float patch_size = m_config.patch * sratio;
            float stride_size = m_config.stride * sratio;
            float posx = ((i - sump) % (pyramid_patches[j].second)) * stride_size;
            float posy = ((i - sump) / (pyramid_patches[j].second)) * stride_size;

            size_t iposx = static_cast<size_t>(posx);
            size_t iposy = static_cast<size_t>(posy);
            size_t posa = std::min(static_cast<size_t>(posx + patch_size),
                    m_config.width);
            size_t posb = std::min(static_cast<size_t>(posy + patch_size),
                    m_config.height);

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

    std::vector<float> FaceFinder::evaluate(const std::string& filepath)
    {
        // load an image (TODO: loads just bitmap image...)
        auto time_point_start = std::chrono::system_clock::now();
        auto image = loadBitmapImage(filepath.c_str());
        auto time_point_finish = std::chrono::system_clock::now();

        if (!image)
        {
            throw FaceFinderException(filepath + " not found");
        }

        std::chrono::duration<double> elapsed_seconds = time_point_finish -
            time_point_start;
        std::cout << "loading test image complete, time: " <<
            elapsed_seconds.count() << "s" << std::endl;

        return evaluate(image);
    }

    std::vector<float> FaceFinder::evaluate(const std::unique_ptr<Image>& image)
    {
        // load image patches from the given image
        std::vector<float> patch_data;
        auto time_point_start = std::chrono::system_clock::now();

        if (m_config.uses_gpu)
        {
            clGetPatches(image, patch_data);
        }
        else
        {
            cpuGetPatches(image, patch_data);
        }

        auto time_point_finish = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = time_point_finish -
            time_point_start;
        std::cout << "loading image patches complete, time: " <<
            elapsed_seconds.count() << "s" << std::endl;
        
        time_point_start = std::chrono::system_clock::now();

        auto category_list = m_network->evaluateAll(patch_data);

        time_point_finish = std::chrono::system_clock::now();
        elapsed_seconds = time_point_finish - time_point_start;
        std::cout << "image patch evaluation complete, time: " <<
            elapsed_seconds.count() << "s" << std::endl;

        size_t positive_res = 0;
        for (auto val: category_list[0])
        {
            if (val == 0)
                positive_res++;
        }

        std::cout << "positive patches = " << positive_res << std::endl;

        time_point_start = std::chrono::system_clock::now();
        auto score = accumulatePatches(category_list[0]);
        time_point_finish = std::chrono::system_clock::now();
        elapsed_seconds = time_point_finish - time_point_start;
        std::cout << "score = " << score << ", score calc time: "
            << elapsed_seconds.count() << std::endl;

        return std::vector<float>{
            static_cast<float>(positive_res),
            static_cast<float>(score)
        };
    }
} // namespace NeuralNet
