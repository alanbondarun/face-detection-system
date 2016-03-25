#include "network.hpp"
#include "image/image.hpp"
#include "calc/calc-cpu.hpp"
#include "netpbm/pm.h"
#include <algorithm>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <ctime>

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

std::unique_ptr<NeuralNet::Image> preprocessImage(
        const std::unique_ptr<NeuralNet::Image>& image)
{
    return NeuralNet::intensityPatch(NeuralNet::leastSquarePatch(
                NeuralNet::equalizePatch(NeuralNet::grayscaleImage(image))));
}

std::vector<float> getRawDataFromImages(
        const std::vector< std::unique_ptr<NeuralNet::Image> >& images)
{
    std::vector<float> data;
    for (auto& img_ptr: images)
    {
        for (size_t c = 0; c < img_ptr->getChannelNum(); c++)
        {
            data.insert(data.end(), img_ptr->getValues(c),
                    img_ptr->getValues(c) +
                        (img_ptr->getWidth() * img_ptr->getHeight()));
        }
    }
    return data;
}

bool load_fddb(std::vector< std::unique_ptr<NeuralNet::Image> >& images,
        size_t num_image, bool uses_first)
{
    const size_t patch_size = 24;
    size_t loaded_image = 0;

    std::ifstream ellipse_list;
    if (uses_first)
    {
        ellipse_list.open("data-fddb/ellipse-1to7.txt");
    }
    else
    {
        ellipse_list.open("data-fddb/ellipse-8to10.txt");
    }

    while (loaded_image < num_image)
    {
        std::string file_line;
        if (!std::getline(ellipse_list, file_line))
        {
            std::cout << "end-of-file of ellipse file" << std::endl;
            return false;
        }
        file_line.insert(file_line.size(), ".ppm");
        file_line.insert(0, "data-fddb/");

        auto img_ptr = NeuralNet::loadPPMImage(file_line.c_str());
        if (!img_ptr)
        {
            std::cout << "missing file " << file_line << std::endl;
            return false;
        }

        std::string line_str;
        if (!std::getline(ellipse_list, line_str))
        {
            std::cout << "end-of-file of ellipse file" << std::endl;
            return false;
        }
        
        std::istringstream iss(line_str);
        size_t face_count;
        iss >> face_count;

        size_t actual_count = 0;
        for (size_t i = 0; i < face_count && actual_count+loaded_image < num_image; i++)
        {
            if (!std::getline(ellipse_list, line_str))
            {
                std::cout << "end-of-file of ellipse file" << std::endl;
                return false;
            }
            
            std::istringstream iss2(line_str);
            float majsize, minsize, tilt, centx, centy;
            iss2 >> majsize >> minsize >> tilt >> centx >> centy;

            if (majsize < patch_size)
                continue;

            auto crop_ptr = NeuralNet::cropImage(img_ptr,
                        centx - majsize/2,
                        centy - minsize/2,
                        majsize,
                        majsize);
            auto fit_ptr = NeuralNet::fitImageTo(crop_ptr, patch_size, patch_size);
            images.push_back(preprocessImage(fit_ptr));
            actual_count++;
        }
        loaded_image += actual_count;
    }

    return true;
}

bool load_faces(std::vector<float>& data, std::vector< std::vector<int> >& category)
{
    const size_t num_image = 2500;
    std::vector< std::unique_ptr<NeuralNet::Image> > images;

    if (!load_fddb(images, num_image, true))
        return false;

    for (auto& gray_ptr: images)
    {
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

bool load_nonface_patch(std::vector< std::unique_ptr<NeuralNet::Image> >& images,
        size_t num_image, size_t train_set)
{
    const size_t set_size = 2000;
    const size_t patch_size = 24;
    const size_t sample_per_img = 10;
    const size_t max_sample_per_img = 1000;
    const float var_thresh = 0.001;
    
    size_t lbound = set_size * train_set + 1;
    size_t ubound = lbound + set_size;

    std::vector<size_t> idxes;
    for (size_t i = lbound; i < ubound; i++)
        idxes.push_back(i);

    std::random_device rd;
    std::mt19937 rgen(rd());
    std::shuffle(idxes.begin(), idxes.end(), rgen);

    size_t num_failed_imgs = 0;
    size_t num_accept_imgs = 0;

    size_t img_count = 0;
    size_t loaded_patch = 0;
    while (loaded_patch < num_image && img_count < idxes.size())
    {
        std::ostringstream oss;
        oss << "image-nonface/img_" << idxes[img_count] << ".bmp";

        auto img_ptr = NeuralNet::loadBitmapImage(oss.str().c_str());
        if (!img_ptr)
        {
            std::cout << "missing file " << oss.str() << std::endl;
            return false;
        }

        std::uniform_int_distribution<size_t> dis_w(0, img_ptr->getWidth() - patch_size);
        std::uniform_int_distribution<size_t> dis_h(0, img_ptr->getHeight() - patch_size);

        size_t added_patch = 0;
        for (size_t i=0; i<max_sample_per_img && added_patch < sample_per_img
                && added_patch + loaded_patch < num_image; i++)
        {
            auto cropImage = NeuralNet::cropImage(
                        img_ptr, dis_w(rgen), dis_h(rgen), patch_size, patch_size);

            auto grayImage = NeuralNet::grayscaleImage(cropImage);

            if (NeuralNet::getVariance(grayImage) <= var_thresh)
            {
                num_failed_imgs++;
            }
            else
            {
                num_accept_imgs++;
            }

            if (NeuralNet::getVariance(grayImage) <= var_thresh)
                continue;

            images.push_back(preprocessImage(cropImage));
            added_patch++;
        }
        loaded_patch += added_patch;

        img_count++;
    }

    std::cout << "accept=" << num_accept_imgs << " reject="
        << num_failed_imgs << std::endl;

    if (loaded_patch == num_image)
        return true;
    return false;
}

bool load_non_faces(std::vector<float>& data, std::vector< std::vector<int> >& category)
{
    const size_t num_image = 11500;

    std::vector< std::unique_ptr<NeuralNet::Image> > images;
    if (!load_nonface_patch(images, num_image, 0))
    {
        return false;
    }

    for (auto& gray_ptr: images)
    {
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
//    const size_t imageCount = 28;
    const size_t test_lfw = 1000;
    const size_t test_fddb = 1000;
    const size_t test_nonface = 1000;

    pm_init(argv[0], 0);

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
/*    std::vector< std::unique_ptr<NeuralNet::Image> > imageList;
    for (size_t i = 1; i <= imageCount; i++)
    {
        std::ostringstream oss;
        oss << "eval-image/" << i << ".bmp";
        imageList.push_back(preprocessImage(
                    NeuralNet::loadBitmapImage(oss.str().c_str())));
    }

    auto originalImgData = getRawDataFromImages(imageList);
    imageList.clear();
    std::cout << "loading evaluation images (original) finished" << std::endl;*/

    std::vector< std::unique_ptr<NeuralNet::Image> > fddbTestImages;
    if (!load_fddb(fddbTestImages, test_fddb, false))
    {
        std::cout << "error loading FDDB test images" << std::endl;
        return false;
    }

    auto fddbTestData = getRawDataFromImages(fddbTestImages);
    fddbTestImages.clear();
    std::cout << "loading evaluation images (FDDB) finished" << std::endl;

    // loading LFW images for evaluation
    std::vector< std::unique_ptr<NeuralNet::Image> > lfwTestImages;
    std::ifstream lfw_name_file("image-lfw/peopleDevTest.txt");
    for (size_t i = 1; i <= test_lfw; i++)
    {
        std::string person_line;
        if (!std::getline(lfw_name_file, person_line))
        {
            std::cout << "end-of-file of LFW names" << std::endl;
            return 0;
        }

        std::istringstream iss(person_line);
        std::string person_name;
        std::getline(iss, person_name, '\t');

        std::string file_name("image-lfw/");
        file_name.append(person_name);
        file_name.append("/");
        file_name.append(person_name);
        file_name.append("_0001.ppm");

        lfwTestImages.push_back(preprocessImage(
                    NeuralNet::loadPPMImage(file_name.c_str())));
    }

    auto lfwTestData = getRawDataFromImages(lfwTestImages);
    lfwTestImages.clear();
    std::cout << "loading evaluation images (LFW) finished" << std::endl;

    std::vector< std::unique_ptr<NeuralNet::Image> > nonFaceImages;
    if (!load_nonface_patch(nonFaceImages, test_nonface, 1))
    {
        std::cout << "error loading evaluation non-face images" << std::endl;
        return 0;
    }

    auto nonFaceTestData = getRawDataFromImages(nonFaceImages);
    nonFaceImages.clear();
    std::cout << "loading evaluation images (non-face) finished" << std::endl;

    // put test sets into the network
    std::vector< std::vector<int> > clist1;
    clist1.emplace_back(test_lfw, 0);
    network.registerTestSet("LFW", std::move(lfwTestData), std::move(clist1));

    std::vector< std::vector<int> > clist2;
    clist2.emplace_back(test_nonface, 1);
    network.registerTestSet("non-face", std::move(nonFaceTestData),
            std::move(clist2));

    std::vector< std::vector<int> > clist3;
    clist3.emplace_back(test_fddb, 0);
    network.registerTestSet("FDDB", std::move(fddbTestData), std::move(clist3));

/*    std::vector< std::vector<int> > clist4;
    std::vector<int> clist4_temp;
    for (size_t i = 0; i < 14; i++)
        clist4_temp.push_back(0);
    for (size_t i = 0; i < 14; i++)
        clist4_temp.push_back(1);
    clist4.push_back(clist4_temp);
    network.registerTestSet("original", std::move(originalImgData),
            std::move(clist4));*/

    std::chrono::time_point<std::chrono::system_clock> time_point_start, time_point_finish;

    if (do_train)
    {
        std::vector<float> data;
        std::vector< std::vector<int> > category;
        category.push_back(std::vector<int>());

        // load face images
        if (!load_faces(data, category))
        {
            return 0;
        }
        std::cout << "loading face images (for training) finished" << std::endl;

        // load non-face images
        if (!load_non_faces(data, category))
        {
            return 0;
        }
        std::cout << "loading non-face images (for training) finished" << std::endl;

        std::cout << "training begin, timer start" << std::endl;
        time_point_start = std::chrono::system_clock::now();

        // actual training goes here
        network.train(data, category);

        time_point_finish = std::chrono::system_clock::now();
        std::cout << "timer finished" << std::endl;
    }
    else
    {
        network.loadFromFiles();

        std::cout << "network weight/bias load complete, timer start" << std::endl;
        time_point_start = std::chrono::system_clock::now();

        network.evaluateTestSets();

        time_point_finish = std::chrono::system_clock::now();
        std::cout << "timer finished" << std::endl;
    }

    std::chrono::duration<double> elapsed_seconds = time_point_finish - time_point_start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;
}
