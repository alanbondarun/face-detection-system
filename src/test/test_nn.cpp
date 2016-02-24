#include "network.hpp"
#include "utils/load_image.hpp"
#include "calc/calc-cpu.hpp"
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>

bool load_test(Json::Value& value)
{
	Json::CharReaderBuilder builder;
	builder["collectComments"] = false;
	
	std::string errors;
	std::fstream dataStream("test.json", std::ios_base::in);
	bool ok = Json::parseFromStream(builder, dataStream, &value, &errors);
	
	if (!ok)
		std::cout << "error while loading test.json: " << errors << std::endl;
	return ok;
}

int main()
{
	const size_t imageCount = 22;
	
	Json::Value networkValue;
	
	if (!load_test(networkValue))
	{
		return 0;
	}
	
	std::vector< std::unique_ptr<NeuralNet::Image> > imageList;
	for (size_t i = 1; i <= imageCount; i++)
	{
		std::ostringstream oss;
		oss << "images/" << i << ".bmp";
		imageList.push_back(std::move(NeuralNet::loadImage(oss.str().c_str())));
	}
	
	std::vector<double> trainData;
	std::vector< std::vector<int> > categoryData;
	for (size_t k = 0; k < imageCount; k++)
	{
		for (size_t i = 0; i < 3; i++)
		{		
			trainData.insert(
					trainData.end(),
					imageList[i]->getValues(i),
					imageList[i]->getValues(i) +
						(imageList[i]->getWidth() * imageList[i]->getHeight())
			);
		}
		if (k < 16)
			categoryData.push_back(std::move(std::vector<int>(1, 1)));
		else
			categoryData.push_back(std::move(std::vector<int>(1, 0)));
	}
	
	NeuralNet::Network network(networkValue);
	network.train(trainData, categoryData);
	network.storeIntoFiles();
}