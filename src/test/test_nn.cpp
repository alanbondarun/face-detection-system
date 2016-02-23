#include "network.hpp"
#include <iostream>
#include <fstream>

int main()
{
	Json::CharReaderBuilder builder;
	builder["collectComments"] = false;
	
	Json::Value dataValue;
	std::string errors;
	std::fstream dataStream("test.json", std::ios_base::in);
	bool ok = Json::parseFromStream(builder, dataStream, &dataValue, &errors);
	if (ok)
	{
		std::cout << "Hello: " << dataValue["hello"] << std::endl;
	}
}