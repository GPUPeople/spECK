#ifndef spECK_BlockRange
#define spECK_BlockRange
#pragma once
#include "limits.cuh"


template<class INDEX_TYPE, class ROW_COUNT_TYPE>
struct BlockRange
{
	// inclusive
	INDEX_TYPE first;
	// if 1, then single row, if > 1 then multiple rows. if 0, then merged with others and must not be used
	ROW_COUNT_TYPE numRows;
	// if nnZ == numeric_limits<COUNT_TYPE>::max(), then this must not be merged with others
	INDEX_TYPE nnz;

	__host__ __device__ BlockRange() : first(spECK::numeric_limits<INDEX_TYPE>::max()), numRows(0), nnz(0) {}

	__host__ __device__ BlockRange(INDEX_TYPE first, INDEX_TYPE numRows, INDEX_TYPE nnz) : first(first)
	{
		this->numRows = min(numRows, spECK::numeric_limits<ROW_COUNT_TYPE>::max());
		this->nnz = min(nnz, spECK::numeric_limits<INDEX_TYPE>::max());
	}

	__host__ __device__ BlockRange& operator=(const BlockRange& a)
	{
		first = a.first;
		numRows = a.numRows;
		nnz = a.nnz;
		return *this;
	}

	__host__ __device__ __forceinline__ INDEX_TYPE nextRow() const { return first + numRows; }
	__host__ __device__ __forceinline__ INDEX_TYPE last() const
	{
		if (numRows == 0)
			return spECK::numeric_limits<INDEX_TYPE>::max();

		return first + numRows - 1;
	}
	__host__ __device__ __forceinline__ bool valid() const { return numRows; }
	__host__ __device__ __forceinline__ void setInvalid() { numRows = 0; }

	__host__ __device__ int operator >(const BlockRange<INDEX_TYPE, ROW_COUNT_TYPE> &b)
	{
		return first > b.first;
	}
};


template<class INDEX_TYPE, class ROW_COUNT_TYPE>
struct BlockRangeKernelScale
{
	// inclusive
	INDEX_TYPE first;
	// if 1, then single row, if > 1 then multiple rows. if 0, then merged with others and must not be used
	ROW_COUNT_TYPE numRows;
	// if nnZ == numeric_limits<COUNT_TYPE>::max(), then this must not be merged with others
	INDEX_TYPE nnz;
	int8_t kernelScale;

	__host__ __device__ BlockRangeKernelScale() : first(spECK::numeric_limits<INDEX_TYPE>::max()), numRows(0), nnz(0), kernelScale(0) {}

	__host__ __device__ BlockRangeKernelScale(INDEX_TYPE first, INDEX_TYPE numRows, INDEX_TYPE nnz, int8_t kernelScale) : first(first), kernelScale(kernelScale)
	{
		this->numRows = min(numRows, spECK::numeric_limits<ROW_COUNT_TYPE>::max());
		this->nnz = min(nnz, spECK::numeric_limits<INDEX_TYPE>::max());
	}

	__host__ __device__ BlockRangeKernelScale& operator=(const BlockRangeKernelScale& a)
	{
		first = a.first;
		numRows = a.numRows;
		nnz = a.nnz;
		kernelScale = a.kernelScale;
		return *this;
	}

	__host__ __device__ __forceinline__ INDEX_TYPE nextRow() const { return first + numRows; }
	__host__ __device__ __forceinline__ INDEX_TYPE last() const
	{
		if (numRows == 0)
			return spECK::numeric_limits<INDEX_TYPE>::max();

		return first + numRows - 1;
	}
	__host__ __device__ __forceinline__ bool valid() const { return numRows; }
	__host__ __device__ __forceinline__ void setInvalid() { numRows = 0; }

	__host__ __device__ int operator >(const BlockRangeKernelScale<INDEX_TYPE, ROW_COUNT_TYPE> &b)
	{
		return first > b.first;
	}
};
#endif