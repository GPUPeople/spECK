#pragma once

#include <string>
#include "CSR.h"
#include "dCSR.h"


template<typename ValueType>
struct Matrices
{
	CSR<ValueType> cpuA, cpuB;
	dCSR<ValueType> gpuA, gpuB;
};

template<typename ValueType>
class DataLoader
{
public:
	DataLoader(std::string path);
	~DataLoader() = default;
	Matrices<ValueType> matrices;
};