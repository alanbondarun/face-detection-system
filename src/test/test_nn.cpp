#include "network.hpp"
#include "image/image.hpp"
#include "calc/calc-cpu.hpp"
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cstring>

bool load_test(Json::Value& value)
{
    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;

    std::string errors;
    std::fstream dataStream("../net_set/test.json", std::ios_base::in);
    bool ok = Json::parseFromStream(builder, dataStream, &value, &errors);

    if (!ok)
        std::cout << "error while loading test.json: " << errors << std::endl;
    return ok;
}

bool load_faces(std::vector<double>& data, std::vector< std::vector<int> >& category)
{
    const size_t num_image = 1000;
    std::ifstream file_name_file("feret-files.out");

    for (size_t i = 1; i <= num_image; i++)
    {
        std::string file_name;
        if (!std::getline(file_name_file, file_name))
        {
            std::cout << "end-of-file of feret-files.out" << std::endl;
            return false;
        }
        file_name.insert(0, "image-feret/");

        auto img_ptr = NeuralNet::loadPPMImage(file_name.c_str());
        if (!img_ptr)
        {
            std::cout << "missing file " << file_name << std::endl;
            return false;
        }

        auto resized_ptr = NeuralNet::fitImageTo(img_ptr, 32, 32);
        for (size_t ch = 0; ch < resized_ptr->getChannelNum(); ch++)
        {
            data.insert(data.end(),
                    resized_ptr->getValues(ch),
                    resized_ptr->getValues(ch) +
                            (resized_ptr->getWidth() * resized_ptr->getHeight()));
        }
        category.push_back(std::move(std::vector<int>({1})));
    }

    return true;
}

bool load_non_faces(std::vector<double>& data, std::vector< std::vector<int> >& category)
{
    const size_t num_image = 1000;

    for (size_t i = 1; i <= num_image; i++)
    {
        std::ostringstream oss;
        oss << "image-nonface/img_" << i << ".bmp";

        auto img_ptr = NeuralNet::loadBitmapImage(oss.str().c_str());
        if (!img_ptr)
        {
            std::cout << "missing file " << oss.str() << std::endl;
            return false;
        }

        auto resized_ptr = NeuralNet::fitImageTo(img_ptr, 32, 32);
        for (size_t ch = 0; ch < resized_ptr->getChannelNum(); ch++)
        {
            data.insert(data.end(),
                    resized_ptr->getValues(ch),
                    resized_ptr->getValues(ch) +
                            (resized_ptr->getWidth() * resized_ptr->getHeight()));
        }
        category.push_back(std::move(std::vector<int>({0})));
    }

    return true;
}

int main(int argc, char* argv[])
{
    const size_t imageCount = 35;

    Json::Value networkValue;

    // load neural net setting
    if (!load_test(networkValue))
    {
        return 0;
    }
    std::cout << "loading settings finished" << std::endl;

    // train only if the argument is given to this program
    bool do_train = false;
    if (argc >= 2 && !strncmp(argv[1], "train", 5))
    {
        do_train = true;
    }

    std::vector<double> data;
    std::vector< std::vector<int> > category;
    if (do_train)
    {
        // load face images
        if (!load_faces(data, category))
        {
            return 0;
        }
        std::cout << "loading face images finished" << std::endl;

        // load non-face images
        if (!load_non_faces(data, category))
        {
            return 0;
        }
        std::cout << "loading non-face images finished" << std::endl;
    }

    // actual training goes here
    NeuralNet::Network network(networkValue);
    if (do_train)
    {
        std::vector< std::unique_ptr<NeuralNet::Image> > imageList;
        for (size_t i = 1; i <= imageCount; i++)
        {
            std::ostringstream oss;
            oss << "eval-image/" << i << ".bmp";
            imageList.push_back(std::move(NeuralNet::loadBitmapImage(oss.str().c_str())));
        }
        std::cout << "loading evaluation images finished" << std::endl;

        const size_t n_epoch = 7;
        for (size_t q = 1; q <= n_epoch; q++)
        {
            network.train(data, category);
            std::cout << "epoch #" << q << " finished, storing infos..."
                    << std::endl;

            network.storeIntoFiles();

            std::cout << "result of epoch #" << q << std::endl;

            // evaluation with other images
            for (size_t i = 0; i < imageCount; i++)
            {
                std::vector<double> eval_data;
                for (size_t j = 0; j < 3; j++)
                {
                    eval_data.insert(eval_data.end(),
                            imageList[i]->getValues(j),
                            imageList[i]->getValues(j) +
                                (imageList[i]->getWidth() * imageList[i]->getHeight())
                    );
                }

                auto res = network.evaluate(eval_data);
                for (auto& category_val: res)
                {
                    std::cout << "image #" << (i+1) << ": " << category_val << std::endl;
                }
            }
        }
    }
    else
    {
        network.loadFromFiles();

        // evaluation with other images
        std::vector< std::unique_ptr<NeuralNet::Image> > imageList;
        for (size_t i = 1; i <= imageCount; i++)
        {
            std::ostringstream oss;
            oss << "eval-image/" << i << ".bmp";
            imageList.push_back(std::move(NeuralNet::loadBitmapImage(oss.str().c_str())));
        }

        for (size_t i = 0; i < imageCount; i++)
        {
            std::vector<double> eval_data;
            for (size_t j = 0; j < 3; j++)
            {
                eval_data.insert(eval_data.end(),
                        imageList[i]->getValues(j),
                        imageList[i]->getValues(j) +
                            (imageList[i]->getWidth() * imageList[i]->getHeight())
                );
            }

            auto res = network.evaluate(eval_data);
            for (auto& category_val: res)
            {
                std::cout << "image #" << (i+1) << ": " << category_val << std::endl;
            }
        }
    }
}
