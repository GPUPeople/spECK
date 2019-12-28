#include "Config.h"
Config *Config::_instance = nullptr;

void Config::init(std::string path)
{
	_instance = new Config(path);
}

void Config::init()
{
	_instance = new Config();
}

int Config::getInt(Key key, int fallback)
{
	if (Instance().overrides.find(key) != Instance().overrides.end())
		return Instance().overrides[key];

	return Instance().reader.GetInteger("", Instance().keyToString[key], fallback);
}

int Config::setInt(Key key, int newVal)
{
	return Instance().overrides[key] = newVal;
}

string Config::getString(Key key, std::string fallback)
{
	return Instance().reader.Get("", Instance().keyToString[key], fallback);
}

bool Config::getBool(Key key, bool fallback)
{
	return Instance().reader.GetBoolean("", Instance().keyToString[key], fallback);
}

float Config::getFloat(Key key, float fallback)
{
	return (float) Instance().reader.GetReal("", Instance().keyToString[key], fallback);
}