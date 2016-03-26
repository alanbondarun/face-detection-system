#ifndef __FACE_FINDER_HPP
#define __FACE_FINDER_HPP

#include <vector>
#include <string>
#include "network.hpp"
#include "image/image.hpp"

namespace NeuralNet
{
    class FaceFinder
    {
    public:
        // exception thrown in this class
        class FaceFinderException: public std::exception
        {
        public:
            FaceFinderException(const std::string& msg)
                : m_msg(std::string("Error at FaceFinder: ") + msg) {}
            virtual ~FaceFinderException() {}
            virtual const char* what() const noexcept { return m_msg.c_str(); }

        private:
            const std::string m_msg;
        };

        FaceFinder(const std::string& config_path);

        // do not allow copying/moving
        FaceFinder(const FaceFinder&) = delete;
        FaceFinder& operator=(const FaceFinder&) = delete;

        std::vector<float> evaluate(const std::string& filepath);
        std::vector<float> evaluate(const std::unique_ptr<Image>& image);

    private:
        struct SearchConfig
        {
            size_t width;
            size_t height;
            size_t patch;
            size_t stride;
            size_t min_image_width;
            float shrink_ratio;
            std::string net_config_file;
            bool grayscale;
            bool uses_gpu;
        };

        // initialization subprocedures
        bool loadConfig(const std::string& filepath);
        void loadNetwork(const std::string& filepath);

        // other subprocedures
        std::vector<cl::Image2D> createImagePyramid(
                cl::Image2D& img_buf);
        cl::Image3D preprocessGrayscalePatches(cl::Image3D& patch_buf);
        std::unique_ptr<Image> preprocessImage(
                const std::unique_ptr<Image>& image);
        void clGetPatches(const std::unique_ptr<Image>& input_img,
                std::vector<float>& patch_data);
        void cpuGetPatches(const std::unique_ptr<Image>& input_img,
                std::vector<float>& patch_data);
        int accumulatePatches(const std::vector<int>& category_list);

        SearchConfig m_config;
        std::unique_ptr<Network> m_network;
    };
}

#endif // __FACE_FINDER_HPP
