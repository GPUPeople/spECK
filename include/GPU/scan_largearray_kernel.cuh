// source https://raw.githubusercontent.com/phebuswink/CUDA/master/MP4/MP4.2/scan_largearray_kernel.cu

#pragma once

#define NUM_BANKS 32U
#define LOG_NUM_BANKS 6U
// MP4.2 - You can use any other block size you wish.
#define BLOCK_SIZE 512U


template<typename T>
struct ParallelScanHostData
{
	ParallelScanHostData() = default;
	T ** BlockSums;
	T ** BlockSumsSummed;
	T ** HostSums;
	T ** HostSumsSummed;
	int * sizes;
};

// includes, kernels
#include <assert.h>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <cub/cub.cuh>

// MP4.2 - Host Helper Functions (allocate your own data structure...)


#define CONFLICT_FREE_OFFSET(n) \
	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

template <typename T, class CombineOp, class InputIterator, class OutputIterator>
__global__ void prescanArrayKernel(OutputIterator out, InputIterator in, int numElements, CombineOp combine)
{

	__shared__ T temp[BLOCK_SIZE * 2 + BLOCK_SIZE / 8];
	int tid = threadIdx.x;

	int start = (BLOCK_SIZE * 2) * blockIdx.x;

	int aj, bj;
	aj = tid;
	bj = tid + BLOCK_SIZE;
	int bankOffsetA = CONFLICT_FREE_OFFSET(aj);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bj);

	if (numElements > start + aj)
	{
		temp[aj + bankOffsetA] = in[start + aj];
	}
	else
	{
		temp[aj + bankOffsetA] = T();
	}
	if (numElements > start + bj)
	{
		temp[bj + bankOffsetB] = in[start + bj];
	}
	else
	{
		temp[bj + bankOffsetB] = T();
	}

	int offset = 1;

#pragma unroll
	for (int d = BLOCK_SIZE; d > 0; d >>= 1)
	{
		__syncthreads();
		if (tid < d)
		{
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] = combine(temp[ai], temp[bi]);

			if (temp[bi].first != temp[ai].first)
			{
				out[start + ai] = temp[ai];
			}
		}
		offset *= 2;
	}
	__syncthreads();

	if (threadIdx.x != 0)
		return;

	if (start + BLOCK_SIZE * 2 - 1 < numElements)
	{
		uint32_t lastId = BLOCK_SIZE * 2 - 1;
		out[start + lastId] = temp[lastId];
	}
}

template <typename T, class CombineOp, class InputIterator, class OutputIterator, uint32_t KERNEL_COUNT>
__global__ void prescanArrayKernelCount(OutputIterator out, InputIterator in, int numElements, CombineOp combine, uint32_t actualKernelCount)
{

	__shared__ T temp[BLOCK_SIZE * 2 + BLOCK_SIZE / 8];
	int tid = threadIdx.x;

	int start = (BLOCK_SIZE * 2) * blockIdx.x;

	int aj, bj;
	aj = tid;
	bj = tid + BLOCK_SIZE;
	int bankOffsetA = CONFLICT_FREE_OFFSET(aj);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bj);

	if (numElements > start + aj)
	{
		temp[aj + bankOffsetA] = in[start + aj];
	}
	else
	{
		temp[aj + bankOffsetA] = T();
	}
	if (numElements > start + bj)
	{
		temp[bj + bankOffsetB] = in[start + bj];
	}
	else
	{
		temp[bj + bankOffsetB] = T();
	}

	int offset = 1;

	uint32_t nonZeroes[KERNEL_COUNT] = {0};

#pragma unroll
	for (int d = BLOCK_SIZE; d > 0; d >>= 1)
	{
		__syncthreads();
		if (tid < d)
		{
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			auto tempA = temp[ai];
			auto combined = combine(tempA, temp[bi]);
			if (combined.first != tempA.first && tempA.first < numElements)
				++nonZeroes[tempA.kernelScale];
			else
				temp[bi] = combined;
		}
		offset *= 2;
	}
	__syncthreads();

	// last one is always saved
	uint32_t lastId = BLOCK_SIZE * 2 - 1;
	if (tid == 0 && temp[lastId].first < numElements)
		++nonZeroes[temp[lastId].kernelScale];

	typedef cub::BlockReduce < uint32_t, BLOCK_SIZE> BlockReduce;
	__shared__ typename BlockReduce::TempStorage tempStorage;

	for (int i = 0; i < actualKernelCount; i++)
	{
		uint32_t thread_storage[1] = {nonZeroes[actualKernelCount - 1 - i]};
		uint32_t sum = BlockReduce(tempStorage).Sum(thread_storage);

		if (threadIdx.x == 0)
		{
			// uint32_t blockStartsIndex = KERNEL_COUNT - actualKernelCount + i;
			int kernelIndex = actualKernelCount - 1 - i;
			uint32_t index = blockIdx.x;
			*(out + numElements * kernelIndex / BLOCK_SIZE + index) = sum;
		}
	}
}

template <typename T, class CombineOp, class InputIterator, class OutputIterator, uint32_t KERNEL_COUNT>
__global__ void prescanArrayKernelNew(InputIterator in, OutputIterator out, uint32_t numElements, 
	CombineOp combine, uint32_t actualKernelCount, uint32_t *offsetCounters)
{

	__shared__ T temp[BLOCK_SIZE * 2 + BLOCK_SIZE / 8];
	__shared__ char prefix[BLOCK_SIZE * 2];
	int tid = threadIdx.x;

	int start = (BLOCK_SIZE * 2) * blockIdx.x;

	int aj, bj;
	aj = tid;
	bj = tid + BLOCK_SIZE;
	int bankOffsetA = CONFLICT_FREE_OFFSET(aj);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bj);

	for (int i = threadIdx.x; i < BLOCK_SIZE * 2; i += BLOCK_SIZE) {
		prefix[i] = 0;
	}

	if (numElements > start + aj)
		temp[aj + bankOffsetA] = in[start + aj];
	else
		temp[aj + bankOffsetA] = T();
	if (numElements > start + bj)
		temp[bj + bankOffsetB] = in[start + bj];
	else
		temp[bj + bankOffsetB] = T();

	int offset = 1;

#pragma unroll
	for (int d = BLOCK_SIZE; d > 0; d >>= 1)
	{
		__syncthreads();
		if (tid < d)
		{
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			auto tempA = temp[ai];
			auto tempB = combine(tempA, temp[bi]);

			if (tempB.first != tempA.first && tempA.first < numElements)
			{
				prefix[ai] = tempA.kernelScale + 1;
			} else
			{
				temp[bi] = tempB;
			}
		}
		offset *= 2;
	}
	__syncthreads();

	uint32_t lastId = BLOCK_SIZE * 2 - 1;
	if (threadIdx.x == BLOCK_SIZE - 1 && temp[lastId].first < numElements)
	{
		prefix[lastId] = temp[lastId].kernelScale + 1;
	}

	typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
	__shared__ typename BlockScan::TempStorage tempStorage;
	__shared__ uint32_t blockOffset;

	int threadIn[2];
	int threadOut[2];

	for (int kernelScale = 0; kernelScale < actualKernelCount; kernelScale++)
	{
		int kernelIndex = actualKernelCount - 1 - kernelScale;

		for (int i = 0; i < 2; i++)
		{
			if (prefix[tid * 2 + i] == kernelScale + 1)
				threadIn[i] = 1;
			else
				threadIn[i] = 0;
		}
		BlockScan(tempStorage).ExclusiveSum(threadIn, threadOut);

		if(threadIdx.x == BLOCK_SIZE - 1)
			blockOffset = atomicAdd(offsetCounters + (KERNEL_COUNT - 1 - kernelScale), threadOut[1] + threadIn[1]) + numElements * kernelIndex;
		__syncthreads();
		
		for (int i = 0; i < 2; i++)
		{
			if (threadIn[i] != 0)  {
				uint32_t index = blockOffset;
				index += threadOut[i];
				auto tmpElement = temp[tid * 2 + i];

				out[index] = toBlockRange(tmpElement.first, tmpElement.numRows);
			}
		}
	}
}

// **===-------- MP4.2 - Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.

template<typename T, class CombineOp>
void prescanArray(T *outArray, T *inArray, int numElements, CombineOp combine)
{
	dim3 dim_block, dim_grid;
	dim_block.x = BLOCK_SIZE;
	dim_block.y = dim_block.z = 1;

	dim_grid.x = ceil((float)(numElements / (float)(dim_block.x * 2)));
	dim_grid.y = dim_grid.z = 1;

	prescanArrayKernel << <dim_grid, dim_block >> > (outArray, inArray, numElements, combine);
}

template <typename T, class InputIterator, class OutputIterator, class CombineOp>
void prescanArray(InputIterator &inputIterator, OutputIterator &outputIterator, CombineOp &combine, int numElements)
{
	dim3 dim_block, dim_grid;
	dim_block.x = BLOCK_SIZE;
	dim_block.y = dim_block.z = 1;

	dim_grid.x = ceil((float)(numElements / (float)(dim_block.x * 2)));
	dim_grid.y = dim_grid.z = 1;

	prescanArrayKernel<T, CombineOp, InputIterator, OutputIterator><<<dim_grid, dim_block>>>(outputIterator, inputIterator, numElements, combine);
}

template <typename T, uint32_t KERNEL_COUNT, class InputIterator, class OutputIterator, class CombineOp>
void prescanArrayOrdered(InputIterator &inputIterator, OutputIterator &outputIterator, 
	CombineOp &combine, uint32_t numElements, uint32_t actualKernelCount, uint32_t *offsetCounters)
{
	dim3 dim_block, dim_grid;
	dim_block.x = BLOCK_SIZE;
	dim_block.y = dim_block.z = 1;

	dim_grid.x = ceil((float)(numElements / (float)(dim_block.x * 2)));
	dim_grid.y = dim_grid.z = 1;

	prescanArrayKernelNew<T, CombineOp, InputIterator, OutputIterator, KERNEL_COUNT><<<dim_grid, dim_block>>>(
		inputIterator, outputIterator, numElements, combine, actualKernelCount, offsetCounters);
}

template<typename T, typename INDEX_TYPE, typename VALUE_TYPE, class InputIterator, class OutputIterator, class CombineOp>
void prescanOperations(InputIterator &operations, OutputIterator &outputIterator, CombineOp &combine, int numElements)
{
	dim3 dim_block, dim_grid;
	dim_block.x = BLOCK_SIZE;
	dim_block.y = dim_block.z = 1;

	dim_grid.x = ceil((float)(numElements / (float)(dim_block.x * 2)));
	dim_grid.y = dim_grid.z = 1;

	prescanArrayKernel<T, CombineOp, InputIterator, OutputIterator> <<<dim_grid, dim_block >>> (outputIterator, operations, numElements, combine);
}

