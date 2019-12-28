#ifndef spECK_Common
#define spECK_Common
#pragma once

template<typename T>
__host__ __device__ __forceinline__ T divup(T a, T b)
{
	return (a + b - 1) / b;
}


template<typename T>
__host__ __device__ __forceinline__ T clamp(const T& a, const T& min, const T& max)
{
	return a < min ? min : (a > max ? max : a);
}
#endif

inline static void HandleError(cudaError_t err,
							   const char *file,
							   int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			   file, line);
		throw std::exception();
	}
}
// #ifdef _DEBUG || NDEBUG || DEBUG
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
// #else
// #define HANDLE_ERROR(err) err
// #endif