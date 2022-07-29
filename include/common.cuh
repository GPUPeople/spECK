#ifndef spECK_Common_2
#define spECK_Common_2
#pragma once


#include "cub/warp/specializations/warp_scan_shfl.cuh"
#include "meta_utils.h"
#include "common.h"
#include <cuda_runtime.h>
#include <type_traits>
#include "GPU/limits.cuh"
#include <device_launch_parameters.h>
#include "GPU/Hash.cuh"
#include "cub/cub.cuh"
#include "WorkDistribution.h"


/////////////// HELPERS /////////////////////////s

template<int END, int BEGIN = 0>
struct ConditionalIteration
{
	template<typename F>
	__device__
		static void iterate(F f)
	{
		bool res = f(BEGIN);
		if (res)
			ConditionalIteration<END, BEGIN + 1>::iterate(f);
	}
};

template<int END>
struct ConditionalIteration<END, END>
{
	template<typename F>
	__device__
		static void iterate(F f)
	{
	}
};


template<int Bytes>
struct VecLoadTypeImpl;

template<>
struct VecLoadTypeImpl<4>
{
	using type = unsigned int;
};
template<>
struct VecLoadTypeImpl<8>
{
	using type = uint2;
};
template<>
struct VecLoadTypeImpl<16>
{
	using type = uint4;
};

template<typename T, int N>
struct VecLoadType
{
	using type = typename VecLoadTypeImpl<sizeof(T)*N>::type;
	union
	{
		T data[N];
		type vec;
	};

	__device__ __forceinline__ VecLoadType() = default;
	__device__ __forceinline__ VecLoadType(type v) : vec(v) {};
};

template<int VecSize, class T, int N>
__device__ __forceinline__ void warp_load_vectorized(T(&out)[N], const T* in)
{
	static_assert(static_popcnt<N>::value == 1, "load_vectorized only works for pow 2 elements");

	using LoadType = VecLoadType<T, VecSize>;
	const typename LoadType::type* vec_in = reinterpret_cast<const typename LoadType::type*>(in + (threadIdx.x / 32)*32*N) + threadIdx.x % 32;

	//TODO: get rid of UB by doing an explicit unroll and just use the vec type
#pragma unroll
	for (int i = 0; i < N / VecSize; ++i)
	{
		LoadType loaded;
		loaded.vec = vec_in[i*32];
#pragma unroll
		for (int j = 0; j < VecSize; ++j)
			out[i*VecSize + j] = loaded.data[j];
	}
}

template<int VecSize, class T, int N>
__device__ __forceinline__ void vectorized_to_blocked(T(&data)[N])
{
	const int Vecs = N / VecSize;

	//rotate
#pragma unroll
	for (int k = 0; k < Vecs - 1; ++k)
	{
		if (threadIdx.x % 32 % Vecs > k)
		{
			T tmp[VecSize];
#pragma unroll
			for (int i = 0; i < VecSize; ++i)
				tmp[i] = data[(Vecs - 1)*VecSize + i];

#pragma unroll
			for (int j = Vecs - 1; j > 0; --j)
#pragma unroll
				for (int i = 0; i < VecSize; ++i)
					data[j*VecSize + i] = data[(j - 1)*VecSize + i];

#pragma unroll
			for (int i = 0; i < VecSize; ++i)
				data[i] = tmp[i];
		}
	}

	//shfl
	int pad_offset = Vecs - (threadIdx.x % 32 * Vecs) / 32;
	int section_offset = (threadIdx.x % 32 * Vecs) % 32;

#pragma unroll
	for (int j = 0; j < Vecs; ++j)
	{
		int shfl_offset = section_offset + ((pad_offset + j) % Vecs);
#pragma unroll
		for (int i = 0; i < VecSize; ++i)
			data[j*VecSize + i] = __shfl(data[j*VecSize + i], shfl_offset);
	}

	//rotate back
#pragma unroll
	for (int k = 0; k < Vecs - 1; ++k)
	{
		if ((threadIdx.x % 32 * Vecs) / 32 > k)
		{
			T tmp[VecSize];
#pragma unroll
			for (int i = 0; i < VecSize; ++i)
				tmp[i] = data[i];

#pragma unroll
			for (int j = 1; j < Vecs; ++j)
#pragma unroll
				for (int i = 0; i < VecSize; ++i)
					data[(j - 1)*VecSize + i] = data[j*VecSize + i];

#pragma unroll
			for (int i = 0; i < VecSize; ++i)
				data[(Vecs - 1)*VecSize + i] = tmp[i];
		}
	}
}


template<class COMP, int LO, int N, int R>
struct ThreadOddEvenMerge;

template<class COMP, int LO, int N, int R, int M, bool FULL>
struct ThreadOddEvenMergeImpl;

template<class T>
__device__ __forceinline__ void swap(T& a, T& b)
{
	T temp = a;
	a = b;
	b = temp;
}

template<class COMP, int LO, int N, int R, int M>
struct ThreadOddEvenMergeImpl<COMP, LO, N, R, M, true>
{
	template<class K, int L>
	__device__ __forceinline__ static void run(K(&key)[L])
	{
		ThreadOddEvenMerge<COMP, LO, N, M>::run(key);
		ThreadOddEvenMerge<COMP, LO + R, N, M>::run(key);
#pragma unroll
		for (int i = LO + R; i + R < LO + N; i += M)
			if (COMP::comp(key[i], key[i + R]))
				swap(key[i], key[i + R]);
	}
	template<class K, class V, int L>
	__device__ __forceinline__ static void run(K(&key)[L], V(&value)[L])
	{
		ThreadOddEvenMerge<COMP, LO, N, M>::run(key, value);
		ThreadOddEvenMerge<COMP, LO + R, N, M>::run(key, value);
#pragma unroll
		for (int i = LO + R; i + R < LO + N; i += M)
			if (COMP::comp(key[i], key[i + R]))
				swap(key[i], key[i + R]),
				swap(value[i], value[i + R]);
	}
};
template<class COMP, int LO, int N, int R, int M>
struct ThreadOddEvenMergeImpl<COMP, LO, N, R, M, false>
{
	template<class K, int L>
	__device__ __forceinline__ static void run(K(&key)[L])
	{
		if (COMP::comp(key[LO], key[LO + R]))
			swap(key[LO], key[LO + R]);
	}
	template<class K, class V, int L>
	__device__ __forceinline__ static void run(K(&key)[L], V(&value)[L])
	{
		if (COMP::comp(key[LO], key[LO + R]))
			swap(key[LO], key[LO + R]),
			swap(value[LO], value[LO + R]);
	}
};


template<class COMP, int LO, int N, int R>
struct ThreadOddEvenMerge : public ThreadOddEvenMergeImpl<COMP, LO, N, R, 2 * R, (2 * R < N)>
{
};

template<class COMP, int LO, int N>
struct ThreadOddEvenMergeSort
{
	template<class K, int L>
	__device__ __forceinline__ static void run(K(&key)[L])
	{
		ThreadOddEvenMergeSort<COMP, LO, N / 2>::run(key);
		ThreadOddEvenMergeSort<COMP, LO + N / 2, N / 2>::run(key);
		ThreadOddEvenMerge<COMP, LO, N, 1>::run(key);
	}
	template<class K, class V, int L>
	__device__ __forceinline__ static void run(K(&key)[L], V(&value)[L])
	{
		ThreadOddEvenMergeSort<COMP, LO, N / 2>::run(key, value);
		ThreadOddEvenMergeSort<COMP, LO + N / 2, N / 2>::run(key, value);
		ThreadOddEvenMerge<COMP, LO, N, 1>::run(key, value);
	}
};

template<class COMP, int LO>
struct ThreadOddEvenMergeSort<COMP, LO, 1>
{
	template<class K, int L>
	__device__ __forceinline__ static void run(K(&key)[L])
	{ }
	template<class K, class V, int L>
	__device__ __forceinline__ static void run(K(&key)[L], V(&value)[L])
	{ }
};

template<class COMP, class K, int L>
__device__ __forceinline__ void threadOddEvenMergeSort(K(&key)[L])
{
	ThreadOddEvenMergeSort<COMP, 0, L>::run(key);
}
template<class COMP, class K, class V, int L>
__device__ __forceinline__ void threadOddEvenMergeSort(K(&key)[L], V(&value)[L])
{
	ThreadOddEvenMergeSort<COMP, 0, L>::run(key, value);
}

struct SortAscending
{
	template<class T>
	__device__ __forceinline__ static bool comp(T a, T b)
	{
		return a > b;
	}
};

struct SortDescending
{
	template<class T>
	__device__ __forceinline__ static bool comp(T a, T b)
	{
		return a < b;
	}
};


// numRows must be > 0 and must not exceed 32!!!
__device__ __host__ __forceinline__ uint32_t toBlockRange(uint32_t startRow, uint32_t numRows)
{
	return (startRow << 5) + (numRows - 1);
}

__device__ __host__ __forceinline__  uint32_t blockRangeToStartRow(uint32_t blockRange)
{
	return blockRange >> 5;
}

__device__ __host__ __forceinline__ uint32_t blockRangeToNumRows(uint32_t blockRange)
{
	return (blockRange & 0b11111) + 1;
}


__device__ __forceinline__ uint32_t toRowColMinMax(uint32_t minCol, uint32_t maxCol)
{
	uint32_t width = 32U - __clz(maxCol - minCol);
	return minCol + (width << 27);
}


__device__ __forceinline__ uint32_t rowColMinMaxtoMinCol(uint32_t rowColMinMax)
{
	return rowColMinMax & ((1 << 27) - 1);
}

__device__ __forceinline__ uint32_t rowColMinMaxtoRowLength(uint32_t rowColMinMax)
{
	// printf("%lu, leads to %lu\n", (rowColMinMax >> 27), 1 << (rowColMinMax >> 27));
	return 1 << (rowColMinMax >> 27);
}

template<typename INDEX_TYPE, typename VALUE_TYPE, class T, uint32_t THREADS>
__global__ void readOperations(dCSRNoDealloc<VALUE_TYPE> matA, dCSRNoDealloc<VALUE_TYPE> matB, T *out, int rowsPerBlock, 
	INDEX_TYPE *maxComputationsPerRow, INDEX_TYPE *rowColMinMax, INDEX_TYPE *rowOperationsMax, INDEX_TYPE *sumProducts)
{
	INDEX_TYPE startRow = blockIdx.x * rowsPerBlock;
	INDEX_TYPE lastRowExcl = min(INDEX_TYPE((blockIdx.x + 1) * rowsPerBlock), INDEX_TYPE(matA.rows));
	bool checkCols = rowColMinMax != nullptr;
	bool checkRowOpsMax = rowOperationsMax != nullptr;
		

	if (startRow >= matA.rows)
		return;

	__shared__ INDEX_TYPE rowOpsCounter[THREADS];
	__shared__ INDEX_TYPE rowOffsets[THREADS];
	__shared__ INDEX_TYPE rowMaxOps[THREADS];
	__shared__ INDEX_TYPE rowMinCols[THREADS];
	__shared__ INDEX_TYPE rowMaxCols[THREADS];
	__shared__ INDEX_TYPE blockProducts;
	__shared__ INDEX_TYPE blockMaxOps;

	rowOpsCounter[threadIdx.x] = 0U;
	rowMaxOps[threadIdx.x] = 0U;
	rowMinCols[threadIdx.x] = spECK::numeric_limits<INDEX_TYPE>::max();
	rowMaxCols[threadIdx.x] = 0U;
	rowOffsets[threadIdx.x] = (startRow + threadIdx.x <= lastRowExcl) ? matA.row_offsets[startRow + threadIdx.x] : matA.nnz;
	if (threadIdx.x == 0) {
		blockProducts = 0;
		blockMaxOps = 0;
	}

	__syncthreads();

	uint32_t startId = rowOffsets[0];
	uint32_t lastIdExcl = lastRowExcl < matA.rows ? rowOffsets[rowsPerBlock] : matA.nnz;

	uint32_t currentRow = spECK::numeric_limits<INDEX_TYPE>::max();
	uint32_t currentRowOps = 0;
	uint32_t currentMin = spECK::numeric_limits<INDEX_TYPE>::max();
	uint32_t currentMax = 0;
	uint32_t currentRowMaxOps = 0;
	for (uint32_t id = threadIdx.x + startId; id < lastIdExcl; id += blockDim.x)
	{
		INDEX_TYPE rowA = 0;

		for(; rowA < rowsPerBlock; ++rowA)
		{
			if (rowOffsets[rowA] <= id && (rowA + startRow + 1 < matA.rows ? rowOffsets[rowA + 1] : matA.nnz) > id)
				break;
		}

		if(currentRow != rowA)
		{
			if (currentRow != spECK::numeric_limits<INDEX_TYPE>::max()) {
				if (checkCols) {
					atomicMin(&rowMinCols[currentRow], currentMin);
					atomicMax(&rowMaxCols[currentRow], currentMax);
				}
				if(checkRowOpsMax)
					atomicMax(&rowMaxOps[currentRow], currentRowMaxOps);
				atomicAdd(&rowOpsCounter[currentRow], currentRowOps);
			}
			currentMin = spECK::numeric_limits<INDEX_TYPE>::max();
			currentMax = 0;
			currentRowMaxOps = 0;
			currentRow = rowA;
			currentRowOps = 0;
		}

		INDEX_TYPE rowB = matA.col_ids[id];
		INDEX_TYPE startIdB = matB.row_offsets[rowB];
		INDEX_TYPE lastIdBExcl = rowB + 1 <= matB.rows ? matB.row_offsets[rowB + 1] : matB.nnz;
		INDEX_TYPE operations = lastIdBExcl - startIdB;

		if(checkCols && startIdB < lastIdBExcl)
		{
			currentMin = min(currentMin, matB.col_ids[startIdB]);
			if (lastIdBExcl > 0)
				currentMax = max(currentMax, matB.col_ids[lastIdBExcl - 1]);
		}

		currentRowOps += operations;
		if(checkRowOpsMax)
			currentRowMaxOps = max(currentRowMaxOps, operations);
	}

	if(currentRow != spECK::numeric_limits<INDEX_TYPE>::max())
	{
		if (checkCols) {
			atomicMin(&rowMinCols[currentRow], currentMin);
			atomicMax(&rowMaxCols[currentRow], currentMax);
		}
		if(checkRowOpsMax)
			atomicMax(&rowMaxOps[currentRow], currentRowMaxOps);
		atomicAdd(&rowOpsCounter[currentRow], currentRowOps);
	}

	__syncthreads();

	if (rowsPerBlock > 1) {
		INDEX_TYPE rowProducts = rowOpsCounter[threadIdx.x];
		for (int i = 16; i > 0; i /= 2)
			rowProducts += __shfl_down_sync(0xFFFFFFFF, rowProducts, i);

		if (threadIdx.x % 32 == 0 && rowProducts > 0)
			atomicAdd(&blockProducts, rowProducts);

		INDEX_TYPE maxRowLength = rowOpsCounter[threadIdx.x];
		for (int i = 16; i > 0; i /= 2)
			maxRowLength = max(maxRowLength, __shfl_down_sync(0xFFFFFFFF, maxRowLength, i));

		if (threadIdx.x % 32 == 0 && maxRowLength > 0)
			atomicMax(&blockMaxOps, maxRowLength);

		__syncthreads();
	}


	if (threadIdx.x < rowsPerBlock && (threadIdx.x + startRow) < matA.rows)
	{
		out[startRow + threadIdx.x] = rowOpsCounter[threadIdx.x];
		if(checkCols)
			rowColMinMax[startRow + threadIdx.x] = toRowColMinMax(rowMinCols[threadIdx.x], rowMaxCols[threadIdx.x]);
		if(checkRowOpsMax)
			rowOperationsMax[startRow + threadIdx.x] = rowMaxOps[threadIdx.x];
	}

	if(threadIdx.x == blockDim.x - 1)
	{
		if (rowsPerBlock == 1) {
			atomicMax(maxComputationsPerRow, rowOpsCounter[0]);
			atomicAdd(sumProducts, rowOpsCounter[0]);
		}
		else {
			atomicMax(maxComputationsPerRow, blockMaxOps);
			atomicAdd(sumProducts, blockProducts);
		}
	}
}

template <typename INDEX_TYPE, uint32_t THREADS, uint32_t rowsPerThreads>
__global__ void getLongestRowA(const INDEX_TYPE* __restrict__ rowOffsets, INDEX_TYPE* __restrict__ longestRow, const INDEX_TYPE rows, const INDEX_TYPE nnz)
{
	typedef cub::BlockReduce<INDEX_TYPE, THREADS> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;

	INDEX_TYPE rowLength[rowsPerThreads];

	for (int i = 0; i < rowsPerThreads; ++i)
		rowLength[i] = 0;

	INDEX_TYPE startRow = blockIdx.x * THREADS * rowsPerThreads + threadIdx.x * rowsPerThreads;
	INDEX_TYPE lastRowExcl = min(rows, blockIdx.x * THREADS * rowsPerThreads + (threadIdx.x + 1) * rowsPerThreads);

	if (lastRowExcl > startRow)
	{
		INDEX_TYPE prevOffset = rowOffsets[startRow];
		for (int i = 1; i <= lastRowExcl - startRow; ++i)
		{
			INDEX_TYPE currentRowOffset = rowOffsets[i + startRow];
			rowLength[i - 1] = currentRowOffset - prevOffset;
			prevOffset = currentRowOffset;
		}
	}

	INDEX_TYPE longestRowBlock = BlockReduce(temp_storage).Reduce(rowLength, cub::Max());

	if (threadIdx.x == 0)
		atomicMax(longestRow, longestRowBlock);
}

template <typename INDEX_TYPE>
__device__ __forceinline__ void markRowSorted(INDEX_TYPE &column)
{
	column |= 1U << 31;
}

template <typename INDEX_TYPE>
__device__ __forceinline__ bool isRowSorted(INDEX_TYPE &column)
{
	return column & (1U << 31);
}

template <typename INDEX_TYPE>
__device__ __forceinline__ void removeSortedMark(INDEX_TYPE &column)
{
	column &= (1U << 31) - 1;
}
template <typename INDEX_TYPE>
__device__ __forceinline__ uint32_t getThreadShiftNew(INDEX_TYPE sumOps, INDEX_TYPE maxOpsPerCol, uint32_t minShift, uint32_t maxShift, INDEX_TYPE cols)
{
	const INDEX_TYPE maxThreads = 1 << maxShift;

	INDEX_TYPE opsPerNnz = max(1U, (sumOps - maxOpsPerCol) / max(1, cols - 1));

	if (opsPerNnz > 64)
		minShift = max(5, minShift);

	INDEX_TYPE shift = max(minShift, min(maxShift, 31 - __clz(opsPerNnz)));
	INDEX_TYPE shift0 = shift;
	if ((1 << shift) * 3 < opsPerNnz * 2 && shift < maxShift)
		++shift;

	INDEX_TYPE shift1 = shift;

	INDEX_TYPE colIters = divup(cols, maxThreads / (1U << shift));
	INDEX_TYPE maxIters = divup(maxOpsPerCol, (1U << shift));

	if (maxIters > colIters * 2)
		shift += min(maxShift - shift, max(1, 31 - __clz(maxIters / colIters / 2)));

	INDEX_TYPE shift2 = shift;
	colIters = divup(cols, maxThreads / (1U << shift));
	maxIters = divup(maxOpsPerCol, (1U << shift));

	if ((1 << shift) * 2 > opsPerNnz && colIters > maxIters * 2)
		shift -= min(shift / 2, max(1, 31 - __clz(colIters / maxIters)));

	INDEX_TYPE shift3 = shift;

	shift = max(minShift, shift);

	INDEX_TYPE shift4 = shift;

	INDEX_TYPE concurrentOps = cols << shift;

	if (concurrentOps < maxThreads)
		shift += 31 - __clz(maxThreads / concurrentOps);

	INDEX_TYPE shift5 = shift;
	shift = max(minShift, min(maxShift, shift));
	INDEX_TYPE shift6 = shift;

	return shift;
}
#endif