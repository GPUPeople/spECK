#include <CUDATools/error.h>
#include <CUDATools/memory.h>
#include <cuda.h>
#include <cuda_runtime.h>


namespace CU
{
	unique_ptr allocMemory(std::size_t size)
	{
		CUdeviceptr ptr;
		cudaMalloc(reinterpret_cast<void**>(&ptr), size);
		return unique_ptr(ptr);
	}
	
	unique_ptr allocMemoryPitched(std::size_t& pitch, std::size_t row_size, std::size_t num_rows, unsigned int element_size)
	{
		CUdeviceptr ptr;
		cudaMallocPitch(reinterpret_cast<void**>(&ptr), &pitch, row_size, num_rows);
		return unique_ptr(ptr);
	}
	
	pitched_memory allocMemoryPitched(std::size_t row_size, std::size_t num_rows, unsigned int element_size)
	{
		CUdeviceptr ptr;
		std::size_t pitch;
		cudaMallocPitch(reinterpret_cast<void**>(&ptr), &pitch, row_size, num_rows);
		return pitched_memory(unique_ptr(ptr), pitch);
	}
}
