#pragma once

#include <string>

class RunConfig
{
public:
	RunConfig(int argc, char *argv[]);
	~RunConfig();
	std::string filePath;
};

