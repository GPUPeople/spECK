
#pragma once

#include <stdint.h>
#include <cuda.h>
#include "Config.h"
#include "GPU/BlockRange.cuh"

class spECKKernels
{
public:
	spECKKernels(uint32_t blockDim=128):
	blockDim{blockDim}
	{}

	void setLaunchDimensions(uint32_t _gridDim, cudaStream_t _stream = 0, uint32_t _blockDim = 128, uint32_t _sharedMem = 0)
	{
		gridDim = _gridDim;
		blockDim = _blockDim;
		stream = _stream;
		sharedMem = _sharedMem;
	}

	// #####################################################################
	// Numeric Hash SpGEMM
	//
	 template <typename INDEX_TYPE, typename VALUE_TYPE, class GlobalMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
	 void h_HashSpGEMMNumeric(dCSRNoDealloc<VALUE_TYPE> matA, dCSRNoDealloc<VALUE_TYPE> matB, dCSRNoDealloc<VALUE_TYPE> matC, GlobalMap *maps, INDEX_TYPE mapCount,
		INDEX_TYPE *blockStartRow, INDEX_TYPE *rowOperations, Config::SortModes sortColumns, uint32_t numberBlocks, const INDEX_TYPE *rowColMinMax,
		INDEX_TYPE *rowMaxOperations, bool setSortedBit, uint32_t rowsPerBlock);

	 // #####################################################################
	 // Numeric SpGEMM Launcher
	 //
	template <typename INDEX_TYPE, typename VALUE_TYPE, class GlobalHashMap, class GlobalRowOffsetMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
	void h_SpGEMMNumericLauncher(dCSRNoDealloc<VALUE_TYPE> matA, dCSRNoDealloc<VALUE_TYPE> matB, dCSRNoDealloc<VALUE_TYPE> matC,
		GlobalHashMap *hashMaps, INDEX_TYPE hashMapCount, GlobalRowOffsetMap *rowOffsetMaps, INDEX_TYPE rowOffsetMapCount,
		INDEX_TYPE *blockStartRow, INDEX_TYPE *rowOperations, Config::SortModes sortColumns, uint32_t numberBlocks, const INDEX_TYPE* rowColMinMax,
		INDEX_TYPE *rowMaxOperations, uint32_t minimumDensity, bool setSortedBit, uint32_t rowsPerBlock);


	 // #####################################################################
	 // Numeric Dense SpGEMM
	 //
	 template <typename INDEX_TYPE, typename VALUE_TYPE, class GlobalMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
	 void h_DenseSpGEMMNumeric(dCSRNoDealloc<VALUE_TYPE> matA, dCSRNoDealloc<VALUE_TYPE> matB, dCSRNoDealloc<VALUE_TYPE> matC, GlobalMap *maps, INDEX_TYPE mapCount,
		INDEX_TYPE *blockStartRow, INDEX_TYPE *rowOperations, uint32_t numberBlocks, const INDEX_TYPE *rowColMinMax,
		INDEX_TYPE *rowMaxOperations, bool setSortedBit, uint32_t rowsPerBlock);

	// #####################################################################
	// Symbolic Dense SpGEMM
	//
	template <typename INDEX_TYPE, typename VALUE_TYPE, class GlobalMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
	void h_DenseSpGEMMCount(dCSRNoDealloc<VALUE_TYPE> matA, dCSRNoDealloc<VALUE_TYPE> matB, GlobalMap *maps, INDEX_TYPE mapCount,
		INDEX_TYPE *matCRowOffsets, INDEX_TYPE *blockStartRow, INDEX_TYPE *rowOperations, uint32_t numberBlocks, const INDEX_TYPE *rowColMinMax,
		INDEX_TYPE *rowMaxOperations, uint32_t *maxNnzPerRow, uint32_t rowsPerBlock);

	 // #####################################################################
	 // Symbolic Hash SpGEMM used for counting NNZ elements of output matrix C
	 //
	 template <typename INDEX_TYPE, typename VALUE_TYPE, unsigned MAX_ROWS_PER_BLOCK, class GlobalMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
	 void h_HashSpGEMMCount(dCSRNoDealloc<VALUE_TYPE> matA, dCSRNoDealloc<VALUE_TYPE> matB, GlobalMap *maps, INDEX_TYPE mapCount, INDEX_TYPE *matCNnzRow,
		 INDEX_TYPE* rowOperations, INDEX_TYPE *blockStartRow, uint32_t numberBlocks, const INDEX_TYPE* rowColMinMax,
		  INDEX_TYPE *rowMaxOperations, uint32_t *maxNnzPerRow, uint32_t rowsPerBlock);

	 // #####################################################################
	 // Symbolic SpGEMM launcher used for counting NNZ elements of output matrix C
	 //
	 template <typename INDEX_TYPE, typename VALUE_TYPE, unsigned MAX_ROWS_PER_BLOCK, class GlobalMap, class GlobalRowOffsetsMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
	 void h_SpGEMMCountLauncher(dCSRNoDealloc<VALUE_TYPE> matA, dCSRNoDealloc<VALUE_TYPE> matB,
								GlobalMap *hashMaps, INDEX_TYPE hashMapCount, GlobalRowOffsetsMap *rowOffsetMaps, INDEX_TYPE rowOffsetMapsCount,
								INDEX_TYPE *matCNnzRow, INDEX_TYPE *rowOperations, INDEX_TYPE *blockStartRow,
								uint32_t numberBlocks, const INDEX_TYPE *rowColMinMax,
								INDEX_TYPE *rowMaxOperations, uint32_t minimumDensity, INDEX_TYPE *maxNnzPerRow, uint32_t rowsPerBlock);

	 // #####################################################################
	 // Sorts results of Symbolic Hash SpGEMM
	 //
	 template <typename INDEX_TYPE, typename VALUE_TYPE, uint32_t THREADS, uint32_t entriesPerBlock>
	 void h_HashSpGEMMSorting(dCSRNoDealloc<VALUE_TYPE> matC, INDEX_TYPE *blockStartRow, uint32_t numberBlocks, bool bitShiftNumRows);

	 template <typename Map, typename INDEX_TYPE, typename VALUE_TYPE>
	 void h_InitializeGlobalMaps(Map *maps, int count, INDEX_TYPE *ids, VALUE_TYPE *values, size_t elementsPerMap);

	 template <typename Map, typename INDEX_TYPE>
	 void h_InitializeGlobalMapsNoVal(Map *maps, int count, INDEX_TYPE *ids, size_t elementsPerMap, uint32_t maxRowsPerBlock);

	 	 
	 // #####################################################################
	 // Load Balancer for HashSpGEMM by assigning blocks to rows -> 1 block can have multiple rows, but 1 row is never shared by multiple blocks
	 // this load balancer works uses the amount of operations per row for balancing
	 //
	 template <typename INDEX_TYPE, typename VALUE_TYPE, typename ROW_COUNT_TYPE, uint8_t KERNEL_COUNT>
	 void h_AssignHashSpGEMMBlocksToRowsOfSameSizeOperations(dCSRNoDealloc<VALUE_TYPE> &matA, dCSRNoDealloc<VALUE_TYPE> &matB, uint32_t *rowOperations,
		 INDEX_TYPE *blockStartRows, INDEX_TYPE *numBlockStarts, INDEX_TYPE (&h_numBlockStarts)[KERNEL_COUNT], INDEX_TYPE *blockStartRowsCombined,
		 uint32_t maxNnzPerBlock, uint32_t maxNnzPerBlockDynamicSharedMem, uint32_t maxRowsPerBlock, uint32_t actualKernelCount, uint32_t &h_rowsRequiringGlobal);



	// #####################################################################
	// Load Balancer for HashSpGEMM by assigning blocks to rows -> 1 block can have multiple rows, but 1 row is never shared by multiple blocks
	// this load balancer tries to combine rows which fit into one as small as possible kernel
	//
	template <typename INDEX_TYPE, typename VALUE_TYPE, typename ROW_COUNT_TYPE, uint8_t KERNEL_COUNT>
	void h_AssignHashSpGEMMBlocksToRowsOfSameSize(dCSRNoDealloc<VALUE_TYPE> &matA,
		INDEX_TYPE *blockStartRows, INDEX_TYPE *blockStartRowsCombined, INDEX_TYPE *numBlockStarts, INDEX_TYPE(&h_numBlockStarts)[KERNEL_COUNT],
		uint32_t maxNnzPerBlock, uint32_t maxNnzPerBlockDynamicSharedMem, uint32_t maxRowsPerBlock, uint32_t actualKernelCount, uint32_t &h_rowsRequiringGlobal);


private:
	uint32_t blockDim;
	uint32_t gridDim;
	uint32_t sharedMem;
	cudaStream_t stream;
};

