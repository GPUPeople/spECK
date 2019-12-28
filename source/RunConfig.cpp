#include "RunConfig.h"

#include <string>
#include <vector>
#include <sstream>
#include "Config.h"

RunConfig::RunConfig(int argc, char *argv[])
{
	if (argc == 1)
	{
		throw "No file path set\n";
	}
		
	filePath = std::string(argv[1]);

	if (argc > 2)
		Config::init(argv[2]);
	else
		Config::init();

	filePath = Config::getString(Config::InputFile, filePath);
}


RunConfig::~RunConfig()
{
}
