#include "network.hpp"
#include "image/image.hpp"
#include "calc/calc-cpu.hpp"
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cstring>
#include <cstdio>

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
    const size_t num_image = 3000;

    std::ifstream file_name_file("feret-files.out");

    for (size_t i = 1; i <= num_image; i++)
    {
        std::unique_ptr<NeuralNet::Image> img_ptr;

        if (i > 1000)
        {
            std::string file_name;
            while (true)
            {
                if (!std::getline(file_name_file, file_name))
                {
                    std::cout << "end-of-file of feret-files.out" << std::endl;
                    return false;
                }
                if (!(file_name.substr(13, 1).compare("f")))
                {
                    break;
                }
            }
            file_name.insert(0, "image-feret/");

            img_ptr = std::move(NeuralNet::loadPPMImage(file_name.c_str()));
            if (!img_ptr)
            {
                std::cout << "missing file " << file_name << std::endl;
                return false;
            }
        }
        else
        {
            char file_cstr[64];
            sprintf(file_cstr, "image-bioid/BioID_%04d.pgm", i);
            img_ptr = std::move(NeuralNet::loadPGMImage(file_cstr));
            if (!img_ptr)
            {
                std::cout << "missing file " << file_cstr << std::endl;
                return false;
            }
        }

//        auto resized_ptr = NeuralNet::fitImageTo(img_ptr, 32, 32);
//        auto gray_ptr = NeuralNet::grayscaleImage(resized_ptr);
        auto gray_ptr = NeuralNet::grayscaleImage(img_ptr);
        for (size_t ch = 0; ch < gray_ptr->getChannelNum(); ch++)
        {
            data.insert(data.end(),
                    gray_ptr->getValues(ch),
                    gray_ptr->getValues(ch) +
                            (gray_ptr->getWidth() * gray_ptr->getHeight()));
        }
        category[0].push_back(1);
        category[0].push_back(0);
    }

    return true;
}

bool load_non_faces(std::vector<double>& data, std::vector< std::vector<int> >& category)
{
    const size_t num_image = 4000;

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

//        auto resized_ptr = NeuralNet::fitImageTo(img_ptr, 32, 32);
//        auto gray_ptr = NeuralNet::grayscaleImage(resized_ptr);
        auto gray_ptr = NeuralNet::grayscaleImage(img_ptr);
        for (size_t ch = 0; ch < gray_ptr->getChannelNum(); ch++)
        {
            data.insert(data.end(),
                    gray_ptr->getValues(ch),
                    gray_ptr->getValues(ch) +
                            (gray_ptr->getWidth() * gray_ptr->getHeight()));
        }
        category[0].push_back(0);
        category[0].push_back(1);
    }

    return true;
}

int main(int argc, char* argv[])
{
    const size_t imageCount = 28;
    const size_t n_eval_ch = 1;

    std::ofstream res_file("result.txt");

    std::cout << "initiating network..." << std::endl;

    Json::Value networkValue;

    // load neural net setting
    if (!load_test(networkValue))
    {
        return 0;
    }
    NeuralNet::Network network(networkValue);
    std::cout << "loading settings finished" << std::endl;

    // train only if the argument is given to this program
    bool do_train = false;
    if (argc >= 2 && !strncmp(argv[1], "train", 5))
    {
        do_train = true;
    }

    // loading images for evaluation
    std::vector< std::unique_ptr<NeuralNet::Image> > imageList;
    for (size_t i = 1; i <= imageCount; i++)
    {
        std::ostringstream oss;
        oss << "eval-image/" << i << ".bmp";
        imageList.push_back(std::move(
            NeuralNet::grayscaleImage(NeuralNet::loadBitmapImage(oss.str().c_str()))
        ));
    }
    std::cout << "loading evaluation images finished" << std::endl;

    if (do_train)
    {
        std::vector<double> data;
        std::vector< std::vector<int> > category;
        category.push_back(std::vector<int>());

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

        const size_t n_epoch = 30;
        for (size_t q = 1; q <= n_epoch; q++)
        {
            // actual training goes here
            network.train(data, category);
            std::cout << "epoch #" << q << " finished, storing infos..."
                    << std::endl;

            network.storeIntoFiles();

            std::cout << "result of epoch #" << q << std::endl;
            std::cout << "(0 = face, 1 = non-face)" << std::endl;
            res_file << "result of epoch #" << q << std::endl;
            res_file << "(0 = face, 1 = non-face)" << std::endl;

            // evaluation with other images
            size_t correct = 0;
            for (size_t i = 0; i < imageCount; i++)
            {
                std::vector<double> eval_data;
                for (size_t j = 0; j < n_eval_ch; j++)
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
                    res_file << "image #" << (i+1) << ": " << category_val << std::endl;
                    if (i<16 && category_val==0)
                        correct++;
                    if (16<=i && i<30 && category_val==1)
                        correct++;
                    if (30<=i && category_val==0)
                        correct++;
                }
            }
            std::cout << "correct answers: " << correct << std::endl;
            res_file << "correct answers: " << correct << std::endl;
        }
    }
    else
    {
        network.loadFromFiles();

        for (size_t i = 0; i < imageCount; i++)
        {
            std::vector<double> eval_data;
            for (size_t j = 0; j < n_eval_ch; j++)
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
