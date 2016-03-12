#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "netpbm/pm.h"
#include "utils/make_unique.hpp"
#include "json/json.h"
#include "image/image.hpp"
#include "image/image_util.hpp"
#include "network.hpp"

struct SearchConfig
{
    size_t width;
    size_t height;
    size_t patch;
    size_t stride;
    size_t min_image_width;
    double shrink_ratio;
    std::string net_config_file;
    std::string test_file;
};

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
    auto image = NeuralNet::loadBitmapImage(config.test_file.c_str());
    std::cout << "loading test image complete" << std::endl;
    
    // load image patches from the given image
    auto image_pyramid = NeuralNet::pyramidImage(image, config.shrink_ratio,
            config.min_image_width);
    std::cout << "loading image pyramid complete" << std::endl;

    std::vector<double> patch_data;
    for (auto& small_image: image_pyramid)
    {
        auto patch_list = NeuralNet::extractPatches(small_image, config.patch,
                config.patch, config.stride);
        for (auto& patch_ptr: patch_list)
        {
            // TODO: patch postprocessing?
            
            auto gray_patch_ptr = NeuralNet::grayscaleImage(patch_ptr);
            auto gray_data = gray_patch_ptr->getValues(0);

            for (size_t i = 0;
                    i < gray_patch_ptr->getWidth() * gray_patch_ptr->getHeight(); i++)
            {
                patch_data.push_back(gray_data[i]);
            }
        }
    }
    std::cout << "loading image patches complete" << std::endl;
    
    // classify patches with the trained network
    auto network = loadNetwork(config.net_config_file);
    if (!network)
    {
        std::cout << "error at loadNetwork()" << std::endl;
        return 0;
    }
    network->loadFromFiles();
    std::cout << "loading network complete" << std::endl;
    
    auto category_list = network->evaluateAll(patch_data);
    std::cout << "image patch evaluation complete" << std::endl;

    size_t positive_res = 0;
    for (auto val: category_list[0])
    {
        if (val == 0)
            positive_res++;
    }

    std::cout << "positive patches = " << positive_res << std::endl;
}
