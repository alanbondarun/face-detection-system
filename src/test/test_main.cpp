#include <iostream>
#include <fstream>
#include <memory>
#include <vector>

#include <unistd.h>

#include "utils/make_unique.hpp"
#include "netpbm/pm.h"
#include "json/json.h"
#include "face_finder.hpp"

struct MainSettingConfig
{
    float patch_num_thresh;
    int vote_num;
    int vote_to_win;
    std::string img_to_eval;
};

int main(int argc, char* argv[])
{
    pm_init(argv[0], 0);

    // load the settings for teh test
    MainSettingConfig config;
    std::string filepath("../net_set/main_set.json");
    try
    {
        Json::CharReaderBuilder builder;
        builder["collectComments"] = false;

        std::fstream settingStream(filepath, std::ios_base::in);

        Json::Value value;
        std::string errors;
        bool ok = Json::parseFromStream(builder, settingStream, &value,
                &errors);
        if (!ok)
        {
            std::cout << "error loading " << filepath << ": " <<
                errors << std::endl;
            return -1;
        }

        config.patch_num_thresh = value["patch_num_thresh"].asFloat();
        config.vote_num = value["vote_num"].asInt();
        config.vote_to_win = value["vote_to_win"].asInt();
        config.img_to_eval = value["img_to_eval"].asString();
    }
    catch (Json::Exception e)
    {
        std::cout << e.what() << std::endl;
        return -1;
    }

    auto finder = std::make_unique<NeuralNet::FaceFinder>(
            "../net_set/search_face.json");

    std::vector<int> votes(config.vote_num, 0);
    int vote_count = 0;
    while (true)
    {
        int vote = 0;
        int image_not_found = 0;
        try
        {
            auto value = finder->evaluate(config.img_to_eval);
            if (value[0] >= config.patch_num_thresh)
            {
                vote = 1;
            }
            else
            {
                vote = 0;
            }
        }
        catch (NeuralNet::FaceFinder::FaceFinderException& e)
        {
            vote = 0;
            image_not_found = 1;
        }

        votes[vote_count] = vote;

        int vote_sum = 0;
        for (auto& v: votes)
        {
            vote_sum += v;
        }

        std::ofstream eval_file("dpms.txt");
        if (vote_sum >= config.vote_to_win || vote == 1)
        {
            eval_file << "1";
            std::cout << "#### TURN ON (vote=" << vote_sum << ") ####\n";
        }
        else
        {
            eval_file << "0";
            std::cout << "**** turn off (vote=" << vote_sum << ") ****\n";
        }
        eval_file.flush();

        if (image_not_found)
            usleep(800000);
        else
            usleep(200000);

        vote_count = (vote_count+1) % config.vote_num;
    }

    return 0;
}
