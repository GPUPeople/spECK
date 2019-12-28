#pragma once

#include <cuda_runtime.h>
#include "dCSR.h"
#include "GPU/spECKKernels.h"
#include "HelperFunctions.cuh"
#include "Config.h"
#include "scan_largearray_kernel.cuh"

template<class INDEX_TYPE, class OUT_TYPE>
class RowLengthReader
{
	INDEX_TYPE *rowOffsets;
	uint32_t rows;
	uint32_t nnz;

public:
	typedef OUT_TYPE value_type;             ///< The type of the element the iterator can point to

	__host__ __device__ __forceinline__ RowLengthReader(INDEX_TYPE *rowOffsets, uint32_t rows, uint32_t nnz)
		: rowOffsets(rowOffsets), rows(rows), nnz(nnz) { }

	__host__ __device__ __forceinline__	OUT_TYPE operator()(const uint32_t &id) const
	{
		// extracts the length of the row
		INDEX_TYPE rowLength = (id + 1 < rows ? rowOffsets[id + 1] : nnz) - rowOffsets[id];
		return OUT_TYPE(id, 1, rowLength);
	}
};

template<class INDEX_TYPE, class OUT_TYPE, uint8_t KERNEL_COUNT>
class RowLengthReaderKernelScale
{
	INDEX_TYPE *rowOffsets;
	uint32_t rows;
	uint32_t nnz;
	uint32_t maxNnzPerBlock;
	uint32_t maxNnzPerBlockDynamicSharedMem;
	uint32_t *rowsLargerShared;

public:
	typedef OUT_TYPE value_type;             ///< The type of the element the iterator can point to

	__host__ __device__ __forceinline__ RowLengthReaderKernelScale(INDEX_TYPE *rowOffsets, uint32_t rows, uint32_t nnz, uint32_t maxNnzPerBlock, 
		uint32_t maxNnzPerBlockDynamicSharedMem, uint32_t* rowsLargerShared)
		: rowOffsets(rowOffsets), rows(rows), nnz(nnz), maxNnzPerBlock(maxNnzPerBlock), 
			maxNnzPerBlockDynamicSharedMem(maxNnzPerBlockDynamicSharedMem), rowsLargerShared(rowsLargerShared) { }

	__device__ __forceinline__	OUT_TYPE operator()(const uint32_t &id) const
	{
		// extracts the length of the row
		uint32_t rowLength = (id + 1 < rows ? rowOffsets[id + 1] : nnz) - rowOffsets[id];

		uint8_t kernelScale = KERNEL_COUNT - 1;
#pragma  unroll
		for (uint32_t i = 1; i < KERNEL_COUNT; ++i)
		{
			// checks if the next lower kernel is still bigger than rowLength
			// i - 1, because this calculation is designed for the 5 kernel sizes that fit into static shared memory, all larger ones go into the largest kernel
			kernelScale -= (rowLength - (maxNnzPerBlock >> (i - 1))) >> 31;
		}

		if (rowLength > maxNnzPerBlockDynamicSharedMem)
			atomicAdd(rowsLargerShared, 1);

		return OUT_TYPE(id, 1, INDEX_TYPE(rowLength), kernelScale);
	}
};

template<class INDEX_TYPE, class OUT_TYPE>
class RowLengthToBlockRange
{
	INDEX_TYPE *rowLengths;

public:
	typedef OUT_TYPE value_type;             ///< The type of the element the iterator can point to

	__host__ __device__ __forceinline__ RowLengthToBlockRange(INDEX_TYPE *rowLengths) : rowLengths(rowLengths) { }

	__host__ __device__ __forceinline__	OUT_TYPE operator()(const uint32_t &id) const
	{
		return OUT_TYPE(id, 1, rowLengths[id]);
	}
};

template<class INDEX_TYPE, class OUT_TYPE, uint8_t KERNEL_COUNT>
class RowLengthToBlockRangeKernelScale
{
	INDEX_TYPE *rowLengths;
	INDEX_TYPE maxNnzPerBlock;
	INDEX_TYPE maxNnzPerBlockDynamicSharedMem;
	INDEX_TYPE *rowCountLongerThanShared;
	INDEX_TYPE maxCols;

public:
	typedef OUT_TYPE value_type;             ///< The type of the element the iterator can point to

	__host__ __device__ __forceinline__ RowLengthToBlockRangeKernelScale(INDEX_TYPE *rowLengths, INDEX_TYPE maxNnzPerBlock, INDEX_TYPE maxNnzPerBlockDynamicSharedMem, INDEX_TYPE *rowCountLongerThanShared, INDEX_TYPE maxCols) :
	rowLengths(rowLengths), maxNnzPerBlock(maxNnzPerBlock), maxNnzPerBlockDynamicSharedMem(maxNnzPerBlockDynamicSharedMem), rowCountLongerThanShared(rowCountLongerThanShared), maxCols(maxCols) { }

	__host__ __device__ __forceinline__	OUT_TYPE operator()(const uint32_t &id) const
	{
		const INDEX_TYPE rowLength = min(rowLengths[id], maxCols);
		uint8_t kernelScale = KERNEL_COUNT - 1;
#pragma  unroll
		for (uint32_t i = 1; i < KERNEL_COUNT; ++i)
		{
			// checks if the next lower kernel is still bigger than rowLength
			// i - 1, because this calculation is designed for the 5 kernel sizes that fit into static shared memory, all larger ones go into the largest kernel
			kernelScale -= (rowLength - (maxNnzPerBlock >> (i - 1))) >> 31;
		}

		if (rowLength > maxNnzPerBlockDynamicSharedMem)
			atomicAdd(rowCountLongerThanShared, 1);

		return OUT_TYPE(id, 1, rowLength, kernelScale);
	}
};

template<class INDEX_TYPE, class ROW_COUNT_TYPE, class IN_TYPE>
class BlockRangeConsumer
{
	INDEX_TYPE *blockStartRows;
	INDEX_TYPE *globalBlockStartCount;

public:
	__host__ __device__ __forceinline__ BlockRangeConsumer(INDEX_TYPE *blockStartRows, INDEX_TYPE *globalBlockStartCount)
		: blockStartRows(blockStartRows), globalBlockStartCount(globalBlockStartCount) { }

	__host__ __device__ __forceinline__
		void operator()(IN_TYPE* virtualOffset, const IN_TYPE value) const
	{
		if (value.first == spECK::numeric_limits<INDEX_TYPE>::max())
			return;

		blockStartRows[value.first] = value.first;
	}
};

template<class INDEX_TYPE, class ROW_COUNT_TYPE, class IN_TYPE, int KERNEL_COUNT>
class BlockRangeConsumerKernelSize
{
	INDEX_TYPE *blockStartRows;
	INDEX_TYPE *globalBlockStartCount;
	INDEX_TYPE entriesPerKernel;
	uint32_t actualKernelCount;

public:
	__host__ __device__ __forceinline__ BlockRangeConsumerKernelSize(INDEX_TYPE *blockStartRows, INDEX_TYPE *globalBlockStartCount, INDEX_TYPE entriesPerKernel, uint32_t actualKernelCount)
		: blockStartRows(blockStartRows), globalBlockStartCount(globalBlockStartCount), entriesPerKernel(entriesPerKernel), actualKernelCount(actualKernelCount) { }

	__host__ __device__ __forceinline__
		void operator()(IN_TYPE* virtualOffset, const IN_TYPE value) const
	{
		if (value.first == spECK::numeric_limits<INDEX_TYPE>::max())
			return;

		int kernelIndex = actualKernelCount - 1 - value.kernelScale;
		INDEX_TYPE index = atomicAdd(globalBlockStartCount + KERNEL_COUNT - 1 - value.kernelScale, 1);

		*(blockStartRows + entriesPerKernel * kernelIndex + index) = toBlockRange(value.first, value.numRows);
	}
};

template <typename INDEX_TYPE>
struct CombineRanges
{
	int maxNnzPerBlock;
	int maxRowsPerBlock;

	CombineRanges(uint32_t maxNnzPerBlock, uint32_t maxRowsPerBlock) :
		maxNnzPerBlock(maxNnzPerBlock), maxRowsPerBlock(maxRowsPerBlock) {}

	template <typename T>
	__host__ __device__ __forceinline__ bool mergeable(const T &a, const T &b) const
	{
		return (a.nnz + b.nnz < maxNnzPerBlock) && (INDEX_TYPE(a.numRows) + INDEX_TYPE(b.numRows) < maxRowsPerBlock);
	}

	template <typename T>
	__host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
	{
		// this is just for empty values
		if (b.first == spECK::numeric_limits<INDEX_TYPE>::max())
			return b;

		if (a.first <= b.first && a.last() >= b.last())
		{
			T retVal = a;
			return retVal;
		}

		if (a.nextRow() == b.first && a.first < b.first && mergeable(a, b))
		{
			T retVal = a;
			retVal.nnz += b.nnz;
			retVal.numRows += b.numRows;
			return retVal;
		}

		return b;
	}
};

template <typename INDEX_TYPE, uint32_t KERNEL_COUNT>
struct CombineRangesOfSameSize
{
	uint32_t maxNnzPerBlock;
	uint32_t maxRowsPerBlock;

	CombineRangesOfSameSize(uint32_t maxNnzPerBlock, uint32_t maxRowsPerBlock) :
		maxNnzPerBlock(maxNnzPerBlock), maxRowsPerBlock(maxRowsPerBlock) {}

	template <typename T>
	__host__ __device__ __forceinline__ bool mergeable(const T &a, const T &b) const
	{
		// - 2 = -1 - 1. first -1, because kernel count is 6, but values go from 0 to 5. 
		// second -1, because maxNnzPerBlock only considers the 5 kernels with static shared 
		// memory size, all larger rows go to the next kernel with dynamic shared mem
		uint32_t maxNnzACurrentScale = maxNnzPerBlock >> (KERNEL_COUNT - 2 - a.kernelScale);
		uint32_t maxNnzBCurrentScale = maxNnzPerBlock >> (KERNEL_COUNT - 2 - b.kernelScale);
		bool isSameScale = a.kernelScale == b.kernelScale && a.kernelScale == 0;

		if (a.numRows + b.numRows > maxRowsPerBlock)
			return false;

		return (isSameScale && (a.nnz + b.nnz < maxNnzACurrentScale));
	}

	template <typename T>
	__host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
	{
		// this is just for empty values
		if (b.first == spECK::numeric_limits<INDEX_TYPE>::max())
			return b;

		if (a.first <= b.first && a.last() >= b.last())
		{
			T retVal = a;
			return retVal;
		}

		if (a.nextRow() == b.first && a.first < b.first && mergeable(a, b))
		{
			T retVal = a;
			retVal.nnz += b.nnz;
			retVal.numRows += b.numRows;
			retVal.kernelScale = max(a.kernelScale, b.kernelScale);
			return retVal;
		}

		return b;
	}
};


struct not_zero
{
	__host__ __device__ __forceinline__	bool operator()(const int x)
	{
		return x != 0;
	}
};

template <typename INDEX_TYPE, typename VALUE_TYPE, typename ROW_COUNT_TYPE, uint8_t KERNEL_COUNT>
void spECKKernels::h_AssignHashSpGEMMBlocksToRowsOfSameSize(dCSRNoDealloc<VALUE_TYPE> &matA,
	INDEX_TYPE *blockStartRows, INDEX_TYPE *blockStartRowsCombined, INDEX_TYPE *numBlockStarts, INDEX_TYPE (&h_numBlockStarts)[KERNEL_COUNT],
	uint32_t maxNnzPerBlock, uint32_t maxNnzPerBlockDynamicSharedMem, uint32_t maxRowsPerBlock, uint32_t actualKernelCount, uint32_t &h_rowsRequiringGlobal)
{
	typedef BlockRangeKernelScale<INDEX_TYPE, ROW_COUNT_TYPE> BlockRangeDef;
	typedef RowLengthReaderKernelScale<INDEX_TYPE, BlockRangeDef, KERNEL_COUNT> RowLengthReaderDef;
	typedef BlockRangeConsumerKernelSize<INDEX_TYPE, ROW_COUNT_TYPE, BlockRangeDef, KERNEL_COUNT> BlockRangeConsumerDef;

	RowLengthReaderDef rowLengthReader(matA.row_offsets, matA.rows, matA.nnz, maxNnzPerBlock, maxNnzPerBlockDynamicSharedMem, blockStartRowsCombined);
	CustomGeneratorIterator<INDEX_TYPE, RowLengthReaderDef> inputIterator(rowLengthReader);

	BlockRangeConsumerDef blockRangeConsumer(blockStartRows, numBlockStarts, matA.rows, actualKernelCount);
	CustomOutputConsumerIterator<BlockRangeDef, BlockRangeConsumerDef> outputIterator(blockRangeConsumer);

	CombineRangesOfSameSize<INDEX_TYPE, KERNEL_COUNT> combineRanges(maxNnzPerBlock, maxRowsPerBlock);

	// first entry is used to find required amount of global maps
	cudaMemset(blockStartRowsCombined, 0, sizeof(INDEX_TYPE));

	prescanArrayOrdered<BlockRangeDef, KERNEL_COUNT>(inputIterator, blockStartRows, combineRanges,
		matA.rows, actualKernelCount, numBlockStarts);

	cudaMemcpy(h_numBlockStarts, numBlockStarts, sizeof(uint32_t) * KERNEL_COUNT, cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_rowsRequiringGlobal, blockStartRowsCombined, sizeof(INDEX_TYPE), cudaMemcpyDeviceToHost);

	uint32_t offset = 0;
	for (int i = 0; i < actualKernelCount; i++)
	{
		int index = KERNEL_COUNT - actualKernelCount + i;

		if (h_numBlockStarts[index] > 0)
		{
			cudaMemcpy(blockStartRowsCombined + offset,
					   blockStartRows + (matA.rows * i),
					   sizeof(uint32_t) * h_numBlockStarts[index],
					   cudaMemcpyDeviceToDevice);
		}
		offset += h_numBlockStarts[index];
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename ROW_COUNT_TYPE, uint8_t KERNEL_COUNT>
void spECKKernels::h_AssignHashSpGEMMBlocksToRowsOfSameSizeOperations(dCSRNoDealloc<VALUE_TYPE> &matA, dCSRNoDealloc<VALUE_TYPE> &matB, uint32_t *rowOperations,
	INDEX_TYPE *blockStartRows, INDEX_TYPE *numBlockStarts, INDEX_TYPE(&h_numBlockStarts)[KERNEL_COUNT], INDEX_TYPE *blockStartRowsCombined,
	uint32_t maxNnzPerBlock, uint32_t maxNnzPerBlockDynamicSharedMem, uint32_t maxRowsPerBlock, uint32_t actualKernelCount, uint32_t &h_rowsRequiringGlobal)
{
	typedef BlockRangeKernelScale<INDEX_TYPE, ROW_COUNT_TYPE> BlockRangeDef;
	typedef RowLengthToBlockRangeKernelScale<INDEX_TYPE, BlockRangeDef, KERNEL_COUNT> RowLengthReaderDef;
	typedef BlockRangeConsumerKernelSize<INDEX_TYPE, ROW_COUNT_TYPE, BlockRangeDef, KERNEL_COUNT> BlockRangeConsumerDef;

	RowLengthReaderDef rowLengthReader(rowOperations, maxNnzPerBlock, maxNnzPerBlockDynamicSharedMem, blockStartRowsCombined, matB.cols);
	CustomGeneratorIterator<INDEX_TYPE, RowLengthReaderDef> inputIterator(rowLengthReader);

	BlockRangeConsumerDef blockRangeConsumer(blockStartRows, numBlockStarts, matA.rows, actualKernelCount);
	CustomOutputConsumerIterator<BlockRangeDef, BlockRangeConsumerDef> outputIterator(blockRangeConsumer);

	CombineRangesOfSameSize<INDEX_TYPE, KERNEL_COUNT> combineRanges(maxNnzPerBlock, maxRowsPerBlock);

	// last entry is used to find required amount of global maps
	cudaMemset(blockStartRowsCombined, 0, sizeof(INDEX_TYPE));

	prescanArrayOrdered<BlockRangeDef, KERNEL_COUNT>(inputIterator, blockStartRows, combineRanges,
													 matA.rows, actualKernelCount, numBlockStarts);

	cudaMemcpy(h_numBlockStarts, numBlockStarts, sizeof(uint32_t) * KERNEL_COUNT, cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_rowsRequiringGlobal, blockStartRowsCombined, sizeof(INDEX_TYPE), cudaMemcpyDeviceToHost);

	uint32_t offset = 0;
	for (int i = 0; i < actualKernelCount; i++)
	{
		int index = KERNEL_COUNT - actualKernelCount + i;

		if (h_numBlockStarts[index] > 0)
		{
			cudaMemcpy(blockStartRowsCombined + offset,
					   blockStartRows + (matA.rows * i),
					   sizeof(uint32_t) * h_numBlockStarts[index],
					   cudaMemcpyDeviceToDevice);
		}
		offset += h_numBlockStarts[index];
	}
}