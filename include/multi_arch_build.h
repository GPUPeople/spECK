#pragma once

#ifdef __CUDACC__
#define DUAL_BUILD_FUNCTION __host__ __device__
#else
#define DUAL_BUILD_FUNCTION 
#endif

#ifndef __CUDA_ARCH__
inline float __uint_as_float(unsigned t)
{
	return *reinterpret_cast<float*>(&t);
}
#endif