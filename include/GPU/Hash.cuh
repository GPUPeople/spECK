#pragma once
#include <cuda_runtime.h>

__host__ __device__ __forceinline__ uint32_t currentHash(uint32_t id) {
	return id * 11;
}