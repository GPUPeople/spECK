// Global includes
#include <bitset>
#include <memory>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Local includes
#include "Multiply.h"
#include "GPU/spECKKernels.h"
#include "GPU/consistent_gpu_memory.h"
#include "CUDATools/stream.h"
#include "meta_utils.h"
#include "GPU/spECK_HashSpGEMM.cuh"
#include "GPU/spECK_HashLoadBalancer.cuh"
#include "GPU/HelperFunctions.cuh"
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include "Config.h"
#include "common.h"
#include "WorkDistribution.h"
#include "HashMap.cuh"
#include "spECKConfig.h"

using IndexType = uint32_t;

namespace spECK
{

template <typename T>
__host__ __forceinline__ T divup(T a, T b)
{
    return (a + b - 1) / b;
}

void startTimerVar(cudaEvent_t &start, CUstream stream = 0)
{
    HANDLE_ERROR(cudaEventRecord(start, stream));
    HANDLE_ERROR(cudaEventSynchronize(start));
}

float recordTimerVar(cudaEvent_t &start, cudaEvent_t &end, CUstream stream = 0)
{
    float time;
    HANDLE_ERROR(cudaEventRecord(end, stream));
    HANDLE_ERROR(cudaEventSynchronize(end));
    HANDLE_ERROR(cudaEventElapsedTime(&time, start, end));
    return time;
}

template <typename DataType, int BLOCKS_PER_SM, int THREADS_PER_BLOCK, int MAX_DYNAMIC_SHARED, int MAX_STATIC_SHARED>
void MultiplyspECKImplementation(const dCSR<DataType> &matA_Dealloc, const dCSR<DataType> &matB_Dealloc, dCSR<DataType> &matOut, spECKConfig &config, Timings &timings)
{
    // those matrices automatically deallocate memory when used as param for cuda -> therefore i have written a new struct without deallocs
    dCSRNoDealloc<DataType> matA(matA_Dealloc), matB(matB_Dealloc);

    if (matB.cols > 1 << 27)
    {
        printf("ERROR: matrix B has more than %d columns (%lu)\n", 1 << 27, matB.cols);
        return;
    }
    if (matA.rows > 1 << 27)
    {
        printf("ERROR: matrix A has more than %d rows (%lu)\n", 1 << 27, matB.rows);
        return;
    }
    if (matA.nnz * matB.nnz == 0) {
        matOut.nnz = 0;
        return;
    }

    if (MAX_DYNAMIC_SHARED != config.maxDynamicSharedMemoryPerBlock || MAX_STATIC_SHARED != config.maxStaticSharedMemoryPerBlock) {
        if (MAX_DYNAMIC_SHARED > config.maxDynamicSharedMemoryPerBlock) {
            printf("ERROR: spECK was compiled with %d maximum dynamic shared memory, but device limit is %d. Please recompile with correct amount set in Multiply.h line 10: spECK_DYNAMIC_MEM_PER_BLOCK\n",
                MAX_DYNAMIC_SHARED, config.maxDynamicSharedMemoryPerBlock);
            return;
        } else {
            printf("WARNING: spECK was compiled with %d maximum dynamic shared memory, but device limit is %d. Please recompile with correct amount set in Multiply.h line 10: spECK_DYNAMIC_MEM_PER_BLOCK\n",
                MAX_DYNAMIC_SHARED, config.maxDynamicSharedMemoryPerBlock);
        }
		if (MAX_STATIC_SHARED > MAX_DYNAMIC_SHARED)
		{
			printf("ERROR: spECK was compiled with smaller dynamic than static shared memory. (%d maximum static shared memory and %d maximum dynamic shared memory). Please check values in Multiply.h line 9 and 10",
				MAX_STATIC_SHARED, MAX_DYNAMIC_SHARED);
			return;
		}
		if (MAX_STATIC_SHARED > config.maxStaticSharedMemoryPerBlock)
		{
			printf("ERROR: spECK was compiled with %d maximum static shared memory, but device limit is %d. Please recompile with correct amount set in Multiply.h line 9: spECK_STATIC_MEM_PER_BLOCK\n",
				MAX_STATIC_SHARED, config.maxStaticSharedMemoryPerBlock);
			return;
		}
		else if (MAX_STATIC_SHARED < config.maxStaticSharedMemoryPerBlock) {
			printf("WARNING: spECK was compiled with %d maximum static shared memory, but device limit is %d. Please recompile with correct amount set in Multiply.h line 9: spECK_STATIC_MEM_PER_BLOCK\n",
				MAX_STATIC_SHARED, config.maxStaticSharedMemoryPerBlock);
		}
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  Constants and configs
    // -------------------------------------------------------------------------------------------------------------------------------------------

	spECKKernels spgemm(1024);

    const int kernelCountNumeric = 6;
    const int kernelCountCounting = 6;
    const int maxRowsPerBlock = 32; // this value may never exceed 32 because of some warp-optimizations
    const int warpsCounting = THREADS_PER_BLOCK / 32;
    const int warpsNumeric = THREADS_PER_BLOCK / 32;
	const int staticSharedMemPerBlockCounting = 48, staticSharedMemPerBlockNumeric = 24;
    const int sharedBytesPerWarpCounting = MAX_STATIC_SHARED / warpsCounting - staticSharedMemPerBlockCounting; // 48 byte is the maximum static shared memory per block
    const int entriesPerWarpCounting = sharedBytesPerWarpCounting / sizeof(IndexType);
	const int sharedBytesPerBlockCounting = sharedBytesPerWarpCounting * warpsCounting;
	// CC version > 7.0 support dynamic shared memory larger than static shared
	const int dynamicSharedBytesPerWarpCounting = MAX_DYNAMIC_SHARED / warpsCounting - staticSharedMemPerBlockCounting; // 48 byte is the maximum static shared memory per block
	const int dynamicEntriesPerWarpCounting = dynamicSharedBytesPerWarpCounting / sizeof(IndexType);
	const int dynamicSharedBytesPerBlockCounting = dynamicSharedBytesPerWarpCounting * warpsCounting;

    const int sharedBytesPerWarpNumeric = MAX_STATIC_SHARED / warpsNumeric - staticSharedMemPerBlockNumeric; // 24 byte is the maximum static shared memory per block
    const int entriesPerWarpNumeric = sharedBytesPerWarpNumeric / (sizeof(IndexType) + sizeof(DataType));
    const int sharedBytesPerBlockNumeric = sharedBytesPerWarpNumeric * warpsNumeric;
	// CC version > 7.0 support dynamic shared memory larger than static shared
    const int dynamicSharedBytesPerWarpNumeric = MAX_DYNAMIC_SHARED / warpsNumeric - staticSharedMemPerBlockNumeric; // 24 byte is the maximum static shared memory per block
	const int dynamicEntriesPerWarpNumeric = dynamicSharedBytesPerWarpNumeric / (sizeof(IndexType) + sizeof(DataType));
	const int dynamicSharedBytesPerBlockNumeric = dynamicSharedBytesPerWarpNumeric * warpsNumeric;
    assert(kernelCountCounting <= kernelCountNumeric);

    bool supportGlobalFallback = true;
    const uint32_t minimumDensityForDenseModeCounting = 999;
    const uint32_t denseModeRowThresholdInternalSorting = 999;
    const uint32_t denseModeRowThresholdExternalSorting = 18;
    const uint32_t sm = config.sm;
    const uint32_t cudaCores = config.sm * BLOCKS_PER_SM * 32;


    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  INITIAL MALLOCS
    // -------------------------------------------------------------------------------------------------------------------------------------------

    int estimatedAvgComPerRow = max(1, int((matA.nnz / matA.rows) * (matB.nnz / matB.rows)));
    // determine how many nnz of matC should be calculated by one block. avoid hashmaps running full
    int maxNnzPerBlockNumeric = entriesPerWarpNumeric * warpsNumeric * 2 / 3;
    int maxNnzPerBlockNumericDynamicSharedMem = dynamicEntriesPerWarpNumeric * warpsNumeric * 2 / 3;

    // CUDA variables
    CUstream stream = config.streams[0];
    auto &streams = config.streams;
    
    if (timings.measureCompleteTime)
        startTimerVar(config.completeStart, stream);

    if (timings.measureAll)
        startTimerVar(config.individualStart, stream);

    // Allocate memory for offsets
    CU::unique_ptr newmat_offsets;
    if (matOut.rows != matA.rows)
    {
        newmat_offsets = CU::allocMemory((matA.rows + 1) * sizeof(IndexType));
    }
    else if (matOut.row_offsets != nullptr)
    {
        newmat_offsets.consume(reinterpret_cast<CUdeviceptr>(matOut.row_offsets));
        matOut.row_offsets = nullptr;
    }

    dCSRNoDealloc<DataType> matC;
    matC.row_offsets = newmat_offsets.get<IndexType>();
    matC.cols = matB.cols;
    matC.rows = matA.rows;

    IndexType *blockStartRowsScale = nullptr;
    IndexType *blockCounterScale = nullptr;
    IndexType h_blockCounterScaleNumeric[kernelCountNumeric] = {0};
    IndexType h_blockCounterScaleCounting[kernelCountCounting] = {0};

    size_t cubTempBytesScan = 0;
    size_t cubTmpBytesReduce = 0;
    size_t cubTmpBytesActual = 0;
    void *cubTmp = nullptr;

    {
        cub::DeviceScan::ExclusiveSum(cubTmp, cubTempBytesScan, matC.row_offsets, matC.row_offsets, matC.rows + 1);
        cub::DeviceReduce::Sum(cubTmp, cubTmpBytesReduce, matC.row_offsets, matC.row_offsets, matC.rows);
        cubTmpBytesReduce = std::max(cubTempBytesScan, cubTmpBytesReduce);
    }

    // ----------------------------------------------------------------------------------

    uint32_t maxComputationsPerRow = 0;
    uint32_t longestRowALength = 0;

    IndexType *d_blockStartRows = nullptr;
    uint32_t *d_blockCounter = nullptr;
    uint32_t *d_rowOperations = nullptr;
    uint32_t *d_rowMaxOperations = nullptr;
    uint32_t *d_maxElementsPerRow = nullptr;
    uint32_t *d_sumProducts = nullptr;
    uint32_t *d_rowColMinMax = nullptr;
    uint32_t *d_maxComputationsPerRow = nullptr;

    uint32_t *d_combined_pointers;
    size_t d_combined_pointers_size = sizeof(uint32_t) * (4 + 2 * matA.rows) + divup(cubTempBytesScan, sizeof(uint32_t)) * sizeof(uint32_t);
    if (matA.nnz > 10000)
        d_combined_pointers_size += sizeof(uint32_t) * matA.rows;

    HANDLE_ERROR(cudaMalloc(&d_combined_pointers, d_combined_pointers_size));
    HANDLE_ERROR(cudaMemsetAsync(d_combined_pointers, 0, d_combined_pointers_size));

    d_maxElementsPerRow = d_combined_pointers;
    /* keep this order */
    d_sumProducts = &d_maxElementsPerRow[1];
    d_maxComputationsPerRow = &d_sumProducts[1];
    /* until here */
    d_blockCounter = &d_maxComputationsPerRow[1];
    d_rowOperations = &d_blockCounter[1];
    d_rowMaxOperations = &d_rowOperations[matA.rows];
    cubTmp = (void *)&d_rowMaxOperations[matA.rows];
    cubTmpBytesActual = cubTempBytesScan;

    if (matA.nnz > 10000)
    {
        d_rowColMinMax = (uint32_t *)cubTmp;
        d_rowColMinMax = &d_rowColMinMax[divup(cubTempBytesScan, sizeof(uint32_t))];
    }

    if (timings.measureAll)
    {
        timings.init = recordTimerVar(config.individualStart, config.individualEnd, stream);
        startTimerVar(config.individualStart, stream);
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  COUNT COMPUTATIONS
    // -------------------------------------------------------------------------------------------------------------------------------------------

    uint32_t sumProducts = 0;
    // calc amount of operations per row
    {
        const uint32_t threadsPerBlock = 128U;
        // limit to threadsPerBlock rows!
        // -> and always try to stay slightly below the threads per block size, because if you are slightly above, it is way more expensive than being far below
        uint32_t rowsPerBlock = std::min(threadsPerBlock, std::max(1U, (threadsPerBlock - 8) / std::max(1U, uint32_t(matA.nnz / matA.rows))));
        rowsPerBlock = std::max(1U, std::min(rowsPerBlock, uint32_t(matA.rows) / (4U * cudaCores / threadsPerBlock)));
        readOperations<IndexType, DataType, IndexType, threadsPerBlock><<<divup(uint32_t(matA.rows), rowsPerBlock), threadsPerBlock>>>(
            matA, matB, d_rowOperations, rowsPerBlock, d_maxComputationsPerRow, d_rowColMinMax, d_rowMaxOperations, d_sumProducts);

        // copying both values at once gives a huge performance boost
        uint32_t tmpArr[2];
        HANDLE_ERROR(cudaMemcpy(&tmpArr, d_sumProducts, sizeof(uint32_t) * 2, cudaMemcpyDeviceToHost));
        sumProducts = tmpArr[0];
        maxComputationsPerRow = tmpArr[1];
        // sumProducts = max(sumProducts, 1);
    }

    if (sumProducts == 0) {
        if (timings.measureCompleteTime)
            timings.complete = recordTimerVar(config.completeStart, config.completeEnd);
        matOut.alloc(matA.rows, matB.cols, 0, false);
        return;
    }

    int maxNnzPerBlockCounting = entriesPerWarpCounting * warpsCounting * 4 / 5;
    int maxNnzPerBlockCountingDynamicSharedMem = dynamicEntriesPerWarpCounting * warpsCounting * 4 / 5;

    // you always know the maximum size of the output row
    uint32_t maxRowLength = max(1, min((uint32_t)matB.cols * 12 / 10, maxComputationsPerRow));

    if (timings.measureAll)
    {
        timings.countProducts = recordTimerVar(config.individualStart, config.individualEnd, stream);
        startTimerVar(config.individualStart, stream);
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  LOADBALANCE COUNTING
    // -------------------------------------------------------------------------------------------------------------------------------------------

    uint32_t h_blockCounter = 0;

    uint32_t rowsPerBlock = 1;
    if (kernelCountCounting > 5 && maxRowLength < (maxNnzPerBlockCounting >> 4)) {
        uint32_t maxRowsPerBlockUtilization = max(1, min(uint32_t(maxRowsPerBlock), uint32_t(matA.rows / (sm * BLOCKS_PER_SM << (kernelCountCounting - 2)))));
        if (maxRowLength < maxNnzPerBlockCounting >> (kernelCountCounting - 1))
        {
            if (estimatedAvgComPerRow / maxRowLength == 1 || maxRowLength / estimatedAvgComPerRow == 1)
                rowsPerBlock = min(maxRowsPerBlockUtilization, ((maxNnzPerBlockCounting >> (kernelCountCounting - 1)) / 3) / maxRowLength);
            else
                rowsPerBlock = min(maxRowsPerBlockUtilization, (maxNnzPerBlockCounting >> kernelCountCounting) / maxRowLength);
        }
        rowsPerBlock = max(rowsPerBlock, 1);
        h_blockCounterScaleCounting[kernelCountCounting - 1] = divup(uint32_t(matA.rows), rowsPerBlock);
    }
    else if (kernelCountCounting > 4 && maxRowLength < (maxNnzPerBlockCounting >> 3))
        h_blockCounterScaleCounting[4] = matA.rows;
    else if (kernelCountCounting > 3 && maxRowLength < (maxNnzPerBlockCounting >> 2))
        h_blockCounterScaleCounting[3] = matA.rows;
    else if (kernelCountCounting > 2 && maxRowLength < (maxNnzPerBlockCounting >> 1))
        h_blockCounterScaleCounting[2] = matA.rows;
    else if (kernelCountCounting > 1 && maxRowLength < (maxNnzPerBlockCounting >> 0))
        h_blockCounterScaleCounting[1] = matA.rows;
    else
        h_blockCounterScaleCounting[0] = matA.rows;
        
    uint32_t rowsRequiringGlobal = h_blockCounterScaleCounting[0];

    uint32_t actualKernelCount = min(kernelCountCounting,
                                     uint32_t(
                                         std::log2(
                                             divup(
                                                 int(maxRowLength),
                                                 min(
                                                     maxNnzPerBlockCounting >> (kernelCountCounting - 1),
                                                     maxNnzPerBlockNumeric >> (kernelCountNumeric - 1)))) +
                                         1));

    bool useLoadBalancingCounting = false;


	// TODO check if && maxComputationsPerRow > maxNnzPerBlockCounting / 8 can be removed
    if (matA.nnz > 771843 || 
        maxComputationsPerRow < maxNnzPerBlockCountingDynamicSharedMem && maxComputationsPerRow > (maxNnzPerBlockCounting >> 2) && matA.rows > 7575 ||
        maxComputationsPerRow > maxNnzPerBlockCountingDynamicSharedMem && sumProducts > 1940177 ||
        maxComputationsPerRow / max(1, int((sumProducts / matA.rows))) > 110 && sumProducts > 1164708)
        useLoadBalancingCounting = true;

    if (useLoadBalancingCounting)
    {
        size_t combinedBlockStartSize = sizeof(IndexType) * (1 + kernelCountCounting + matA.rows * (1 + actualKernelCount));

        HANDLE_ERROR(cudaMalloc(&d_blockStartRows, combinedBlockStartSize));
        blockStartRowsScale = &d_blockStartRows[matA.rows + 1];
        blockCounterScale = &blockStartRowsScale[actualKernelCount * matA.rows];
        HANDLE_ERROR(cudaMemset(blockCounterScale, 0, sizeof(IndexType) * kernelCountCounting));

        // load balance over amount of operations per row in A
        spgemm.h_AssignHashSpGEMMBlocksToRowsOfSameSizeOperations<uint32_t, DataType, uint8_t, kernelCountCounting>(
            matA, matB, d_rowOperations, blockStartRowsScale, blockCounterScale, h_blockCounterScaleCounting, d_blockStartRows,
            maxNnzPerBlockCounting, maxNnzPerBlockCountingDynamicSharedMem, maxRowsPerBlock, actualKernelCount, rowsRequiringGlobal);
    }
    else
    {
        h_blockCounter = matA.rows;
        d_blockStartRows = nullptr;
    }

    if (timings.measureAll)
    {
        timings.loadBalanceCounting = recordTimerVar(config.individualStart, config.individualEnd, stream);
        startTimerVar(config.individualStart, stream);
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  ALLOCATE GLOBAL MAPS
    // -------------------------------------------------------------------------------------------------------------------------------------------

    int elementsPerMap = (std::max(maxRowLength, uint32_t(maxNnzPerBlockCountingDynamicSharedMem)) * 5) / 4;
    supportGlobalFallback &= maxRowLength > entriesPerWarpCounting * warpsCounting;

    typedef HashMap<uint32_t, DataType> GlobalMap;
    typedef HashMapNoValue<uint32_t, 1> GlobalMapRowOffsets;
    typedef HashMapNoValue<uint32_t, maxRowsPerBlock> GlobalMapNoValue;
    void *hashMaps = nullptr;
    IndexType *maps_indices = nullptr;
    DataType *maps_values = nullptr;
    uint32_t hashMapCount = 0;
    size_t globalMapMaxSize;
    globalMapMaxSize = std::max(sizeof(GlobalMap), sizeof(GlobalMapNoValue));
    globalMapMaxSize = std::max(globalMapMaxSize, sizeof(GlobalMapRowOffsets));

    if (supportGlobalFallback)
    {
        hashMapCount = std::min(sm * BLOCKS_PER_SM, h_blockCounterScaleCounting[0]);
        hashMapCount = std::min(hashMapCount, rowsRequiringGlobal);
        supportGlobalFallback &= hashMapCount > 0;
    }

    rowsRequiringGlobal = matB.cols < entriesPerWarpCounting * warpsCounting ? 0 : rowsRequiringGlobal;
    bool isDenseCounting = useLoadBalancingCounting && rowsRequiringGlobal > 0 && maxComputationsPerRow > maxNnzPerBlockCountingDynamicSharedMem * 2;

    if (isDenseCounting)
    {
        supportGlobalFallback = false;
        // every bit is one column
        if (matB.cols > (warpsCounting * sharedBytesPerWarpCounting * 8) / 2)
        {
            if (longestRowALength == 0)
            {
                uint32_t *d_longestRowALength = nullptr;
                HANDLE_ERROR(cudaMalloc(&d_longestRowALength, sizeof(uint32_t)));
                HANDLE_ERROR(cudaMemset(d_longestRowALength, 0, sizeof(uint32_t)));

                const uint32_t blockdim = 256;
                const uint32_t rowsPerThread = 2;
                const uint32_t blocks = divup(IndexType(matA.rows), blockdim * rowsPerThread);
                getLongestRowA<IndexType, blockdim, rowsPerThread><<<blocks, blockdim>>>(matA.row_offsets, d_longestRowALength, matA.rows, matA.nnz);
                cudaMemcpy(&longestRowALength, d_longestRowALength, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            }

            // only use global maps if the row cursors can't be held in shared memory
            if (elementsPerMap * 2 > warpsCounting * entriesPerWarpCounting)
            {
                hashMapCount = sm * BLOCKS_PER_SM;
                elementsPerMap = longestRowALength * 5 / 4;

                if (maps_indices != nullptr)
                    HANDLE_ERROR(cudaFree(maps_indices));
                if (hashMaps != nullptr)
                    HANDLE_ERROR(cudaFree(hashMaps));

                HANDLE_ERROR(cudaMalloc(&maps_indices, sizeof(uint32_t) * hashMapCount * (elementsPerMap + maxRowsPerBlock + 1)));
                HANDLE_ERROR(cudaMalloc(&hashMaps, globalMapMaxSize * hashMapCount));

                spgemm.setLaunchDimensions(hashMapCount, streams[0], 32 * warpsNumeric);
                spgemm.h_InitializeGlobalMapsNoVal<GlobalMapRowOffsets, uint32_t>((GlobalMapRowOffsets *)hashMaps, hashMapCount, maps_indices, elementsPerMap, maxRowsPerBlock);
            }
        }
    }

    if (supportGlobalFallback)
    {
        HANDLE_ERROR(cudaMalloc(&hashMaps, globalMapMaxSize * hashMapCount));
        HANDLE_ERROR(cudaMalloc(&maps_indices, sizeof(IndexType) * hashMapCount * (elementsPerMap + maxRowsPerBlock + 1)));

        spgemm.setLaunchDimensions(hashMapCount, streams[0], 32 * warpsCounting);
        spgemm.h_InitializeGlobalMapsNoVal<GlobalMapNoValue, IndexType>((GlobalMapNoValue *)hashMaps, hashMapCount, maps_indices, elementsPerMap, maxRowsPerBlock);
    }

    if (timings.measureAll)
    {
        timings.globalMapsCounting = recordTimerVar(config.individualStart, config.individualEnd, stream);
        startTimerVar(config.individualStart, stream);
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  PRE-COUNTING LOAD-OPTIMIZATION
    // -------------------------------------------------------------------------------------------------------------------------------------------

    IndexType blockPrefixScaled[kernelCountCounting] = {0};
    {
        uint32_t activeSM = h_blockCounterScaleCounting[0];
        // never go up to top level
        int firstXEmpty = h_blockCounterScaleCounting[0] == 0;
        bool foundFirstNonEmpty = h_blockCounterScaleCounting[0] != 0;
        for (int i = 1; i < kernelCountCounting; ++i)
        {
            blockPrefixScaled[i] = h_blockCounterScaleCounting[i - 1] + blockPrefixScaled[i - 1];
            activeSM += 2 * h_blockCounterScaleCounting[i] >> (i - 1);
            if (!foundFirstNonEmpty)
            {
                if (h_blockCounterScaleCounting[i] == 0)
                    firstXEmpty++;
                else
                    foundFirstNonEmpty = true;
            }
        }

        // avoid div by zero
        activeSM = max(activeSM, 1);

        if (activeSM < sm * BLOCKS_PER_SM)
        {
            int shiftUp = min(firstXEmpty, int(std::log2(sm * BLOCKS_PER_SM / activeSM)));

            if (shiftUp > 0)
            {
                for (int i = 0; i < kernelCountCounting; i++)
                {
                    if (i + shiftUp < kernelCountCounting)
                    {
                        h_blockCounterScaleCounting[i] = h_blockCounterScaleCounting[i + shiftUp];
                        blockPrefixScaled[i] = blockPrefixScaled[i + shiftUp];
                    }
                    else
                    {
                        h_blockCounterScaleCounting[i] = 0;
                        blockPrefixScaled[i] = h_blockCounter;
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  COUNT NNZ PER ROW OF C
    // -------------------------------------------------------------------------------------------------------------------------------------------

    {
        if (h_blockCounterScaleCounting[0] > 0)
        {
            if (isDenseCounting)
            {
                // this only uses 1 block per sm and therefore hash 50% occupancy, but better caching
                spgemm.setLaunchDimensions(h_blockCounterScaleCounting[0], streams[0], (32 * warpsCounting >> 0), dynamicSharedBytesPerBlockCounting);
                spgemm.h_DenseSpGEMMCount<IndexType, DataType, GlobalMapRowOffsets, dynamicSharedBytesPerBlockCounting, true, (32 * warpsCounting >> 0)>(
                    matA, matB, (GlobalMapRowOffsets *)hashMaps, hashMapCount, matC.row_offsets, d_blockStartRows + blockPrefixScaled[0],
                    d_rowOperations, h_blockCounterScaleCounting[0], d_rowColMinMax,
                    d_rowMaxOperations, d_maxElementsPerRow, rowsPerBlock);
            }
            else
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleCounting[0], streams[0], 32 * warpsCounting >> 0, dynamicSharedBytesPerBlockCounting);
                spgemm.h_SpGEMMCountLauncher<IndexType, DataType, maxRowsPerBlock, GlobalMapNoValue, GlobalMapRowOffsets, dynamicSharedBytesPerBlockCounting, true, (32 * warpsCounting >> 0)>(
                    matA, matB, (GlobalMapNoValue *)hashMaps, hashMapCount, nullptr, 0, matC.row_offsets, d_rowOperations,
                    d_blockStartRows + blockPrefixScaled[0], h_blockCounterScaleCounting[0], d_rowColMinMax,
                    d_rowMaxOperations, minimumDensityForDenseModeCounting, d_maxElementsPerRow, rowsPerBlock);
            }
        }

        if (kernelCountCounting > 1 && h_blockCounterScaleCounting[1] > 0)
        {
            spgemm.setLaunchDimensions(h_blockCounterScaleCounting[1], streams[1], 32 * warpsCounting >> 0, sharedBytesPerBlockCounting >> 0);
            spgemm.h_SpGEMMCountLauncher<IndexType, DataType, maxRowsPerBlock, GlobalMapNoValue, GlobalMapRowOffsets, (sharedBytesPerBlockCounting >> 0), false, (32 * warpsCounting >> 0)>(
                matA, matB, (GlobalMapNoValue *)hashMaps, hashMapCount, nullptr, 0, matC.row_offsets, d_rowOperations,
                d_blockStartRows + blockPrefixScaled[1], h_blockCounterScaleCounting[1], d_rowColMinMax,
                d_rowMaxOperations, minimumDensityForDenseModeCounting, d_maxElementsPerRow, rowsPerBlock);
        }

        if (kernelCountCounting > 2 && h_blockCounterScaleCounting[2] > 0)
        {
            spgemm.setLaunchDimensions(h_blockCounterScaleCounting[2], streams[2], (32 * warpsCounting >> 1), sharedBytesPerBlockCounting >> 1);
            spgemm.h_SpGEMMCountLauncher<IndexType, DataType, maxRowsPerBlock, GlobalMapNoValue, GlobalMapRowOffsets, (sharedBytesPerBlockCounting >> 1), false, (32 * warpsCounting >> 1)>(
                matA, matB, (GlobalMapNoValue *)hashMaps, hashMapCount, nullptr, 0, matC.row_offsets, d_rowOperations,
                d_blockStartRows + blockPrefixScaled[2], h_blockCounterScaleCounting[2], d_rowColMinMax,
                d_rowMaxOperations, minimumDensityForDenseModeCounting, d_maxElementsPerRow, rowsPerBlock);
        }

        if (kernelCountCounting > 3 && h_blockCounterScaleCounting[3] > 0)
        {
            spgemm.setLaunchDimensions(h_blockCounterScaleCounting[3], streams[3], (32 * warpsCounting >> 2), sharedBytesPerBlockCounting >> 2);
            spgemm.h_SpGEMMCountLauncher<IndexType, DataType, maxRowsPerBlock, GlobalMapNoValue, GlobalMapRowOffsets, (sharedBytesPerBlockCounting >> 2), false, (32 * warpsCounting >> 2)>(
                matA, matB, (GlobalMapNoValue *)hashMaps, hashMapCount, nullptr, 0, matC.row_offsets, d_rowOperations,
                d_blockStartRows + blockPrefixScaled[3], h_blockCounterScaleCounting[3], d_rowColMinMax,
                d_rowMaxOperations, minimumDensityForDenseModeCounting, d_maxElementsPerRow, rowsPerBlock);
        }

        if (kernelCountCounting > 4 && h_blockCounterScaleCounting[4] > 0)
        {
            spgemm.setLaunchDimensions(h_blockCounterScaleCounting[4], streams[4], 32 * warpsCounting >> 3, sharedBytesPerBlockCounting >> 3);
            spgemm.h_SpGEMMCountLauncher<IndexType, DataType, maxRowsPerBlock, GlobalMapNoValue, GlobalMapRowOffsets, (sharedBytesPerBlockCounting >> 3), false, (32 * warpsCounting >> 3)>(
                matA, matB, (GlobalMapNoValue *)hashMaps, hashMapCount, nullptr, 0, matC.row_offsets, d_rowOperations,
                d_blockStartRows + blockPrefixScaled[4], h_blockCounterScaleCounting[4], d_rowColMinMax,
                d_rowMaxOperations, minimumDensityForDenseModeCounting, d_maxElementsPerRow, rowsPerBlock);
        }

        if (kernelCountCounting > 5 && h_blockCounterScaleCounting[5] > 0)
        {
            spgemm.setLaunchDimensions(h_blockCounterScaleCounting[5], streams[5], 32 * warpsCounting >> 4, sharedBytesPerBlockCounting >> 4);
            spgemm.h_SpGEMMCountLauncher<IndexType, DataType, maxRowsPerBlock, GlobalMapNoValue, GlobalMapRowOffsets, (sharedBytesPerBlockCounting >> 4), false, (32 * warpsCounting >> 4)>(
                matA, matB, (GlobalMapNoValue *)hashMaps, hashMapCount, nullptr, 0, matC.row_offsets, d_rowOperations,
                d_blockStartRows + blockPrefixScaled[5], h_blockCounterScaleCounting[5], d_rowColMinMax,
                d_rowMaxOperations, minimumDensityForDenseModeCounting, d_maxElementsPerRow, rowsPerBlock);
        }
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  SCAN ROW OFFSETS AND GET NNZ OF C
    // -------------------------------------------------------------------------------------------------------------------------------------------

    // now we need to allocate that memory for prefix scan and for finding the longest row
    if (cubTmpBytesActual < cubTempBytesScan)
    {
        cubTmpBytesActual = cubTempBytesScan;
        if (cubTmp != nullptr)
            HANDLE_ERROR(cudaFree(cubTmp));
        HANDLE_ERROR(cudaMalloc(&cubTmp, cubTmpBytesActual));
    }

    // prefix sum to get the starting ids of each row of mat C
    cub::DeviceScan::ExclusiveSum(cubTmp, cubTmpBytesActual, matC.row_offsets, matC.row_offsets, matC.rows + 1);
    {
        IndexType nnz;
        cudaMemcpy(&nnz, matC.row_offsets + matC.rows, sizeof(IndexType), cudaMemcpyDeviceToHost);
        matC.nnz = nnz;
    }

    if (timings.measureAll)
    {
        HANDLE_ERROR(cudaDeviceSynchronize());
        timings.spGEMMCounting = recordTimerVar(config.individualStart, config.individualEnd, stream);
        startTimerVar(config.individualStart, stream);
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  ALLOCATE OUTPUT MATRIX C
    // -------------------------------------------------------------------------------------------------------------------------------------------

    // only allocate mem for mat C if size is not correct
    if (matOut.nnz != matC.nnz)
    {
        matOut.alloc(matC.rows, matC.cols, matC.nnz, false);
    }

    if (matOut.data == nullptr || matOut.col_ids == nullptr)
    {
        if (matOut.nnz > 0)
            printf("ERROR: out of memory\n");
        return;
    }

    matOut.row_offsets = std::move(newmat_offsets.getRelease<IndexType>());
    matC = dCSRNoDealloc<DataType>(matOut);

    if (timings.measureAll)
    {
        timings.allocC = recordTimerVar(config.individualStart, config.individualEnd, stream);
        startTimerVar(config.individualStart, stream);
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  LOAD BALANCE NUMERIC
    // -------------------------------------------------------------------------------------------------------------------------------------------

    uint32_t maxElementsPerRow = maxRowLength;
    cudaMemcpy(&maxElementsPerRow, d_maxElementsPerRow, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    bool reprocessLoadBalanceNumeric = useLoadBalancingCounting;
    rowsPerBlock = 1;
    
    // get the longest row in order to minimize the global map size which needs to be allocated

    if (kernelCountNumeric > 5 && maxElementsPerRow < (maxNnzPerBlockNumeric >> 4)) {
        uint32_t maxRowsPerBlockUtilization = max(1, min(uint32_t(maxRowsPerBlock), uint32_t(matA.rows / (sm * BLOCKS_PER_SM << (kernelCountNumeric - 2)))));
        if (maxElementsPerRow<(entriesPerWarpNumeric * warpsNumeric)>> kernelCountNumeric)
        {
            if (maxElementsPerRow / max(1U, uint32_t(matC.nnz / matC.rows)) == 1)
                rowsPerBlock = min(maxRowsPerBlockUtilization, (maxNnzPerBlockNumeric >> (kernelCountNumeric - 1)) / maxElementsPerRow);
            else
                rowsPerBlock = min(maxRowsPerBlockUtilization, (entriesPerWarpNumeric * warpsNumeric >> (kernelCountNumeric - 1)) / maxElementsPerRow);
        }
        rowsPerBlock = max(rowsPerBlock, 1);
        h_blockCounterScaleNumeric[kernelCountNumeric - 1] = divup(uint32_t(matA.rows), rowsPerBlock);
    }
    else if (kernelCountNumeric > 4 && maxElementsPerRow < (maxNnzPerBlockNumeric >> 3))
        h_blockCounterScaleNumeric[4] = matC.rows;
    else if (kernelCountNumeric > 3 && maxElementsPerRow < (maxNnzPerBlockNumeric >> 2))
        h_blockCounterScaleNumeric[3] = matC.rows;
    else if (kernelCountNumeric > 2 && maxElementsPerRow < (maxNnzPerBlockNumeric >> 1))
        h_blockCounterScaleNumeric[2] = matC.rows;
    else if (kernelCountNumeric > 1 && maxElementsPerRow < (maxNnzPerBlockNumeric >> 0))
        h_blockCounterScaleNumeric[1] = matC.rows;
    else
        h_blockCounterScaleNumeric[0] = matC.rows;

    supportGlobalFallback = true;
    supportGlobalFallback &= maxElementsPerRow >= maxNnzPerBlockNumericDynamicSharedMem;
    rowsRequiringGlobal = h_blockCounterScaleNumeric[0];

    uint32_t avgElementsPerRow = max(1, int(matC.nnz / matC.rows));
    uint32_t maxAvgElementsPerRowRatio = maxElementsPerRow / avgElementsPerRow;
    reprocessLoadBalanceNumeric = false;
    if (maxElementsPerRow > (maxNnzPerBlockNumeric >> 2) && matA.rows >= 1236 && sumProducts > 636293 ||
        maxElementsPerRow > (maxNnzPerBlockNumeric >> (kernelCountNumeric - 1)) && (
            maxAvgElementsPerRowRatio > 4 && sumProducts > 4921876 ||
            maxAvgElementsPerRowRatio > 13 && sumProducts > 385847 ||
            maxAvgElementsPerRowRatio > 18 && sumProducts > 26263 && avgElementsPerRow > 22 ||
            maxAvgElementsPerRowRatio > 146))
        reprocessLoadBalanceNumeric = true;

    // can bring a performance benefit for some matrices, but has small overhead
    if (reprocessLoadBalanceNumeric && matC.nnz > 0)
    {
        if (d_blockCounter == nullptr)
        {
            HANDLE_ERROR(cudaMalloc(&d_blockCounter, sizeof(uint32_t)));
        }
        if (blockCounterScale == nullptr)
        {
            size_t combinedBlockStartSize = sizeof(IndexType) * (1 + kernelCountNumeric + matA.rows * (1 + actualKernelCount));

            HANDLE_ERROR(cudaMalloc(&d_blockStartRows, combinedBlockStartSize));
            blockStartRowsScale = &d_blockStartRows[matA.rows + 1];
            blockCounterScale = &blockStartRowsScale[actualKernelCount * matA.rows];
        }
        // reset buffers
        HANDLE_ERROR(cudaMemsetAsync(d_blockCounter, 0, sizeof(uint32_t)));
        HANDLE_ERROR(cudaMemsetAsync(blockCounterScale, 0, sizeof(IndexType) * kernelCountNumeric));

        spgemm.h_AssignHashSpGEMMBlocksToRowsOfSameSize<IndexType, DataType, uint8_t, kernelCountNumeric>(
            matC, blockStartRowsScale, d_blockStartRows, blockCounterScale, h_blockCounterScaleNumeric,
            maxNnzPerBlockNumeric, maxNnzPerBlockNumericDynamicSharedMem, maxRowsPerBlock, actualKernelCount, rowsRequiringGlobal);
    }
    else
    {
        HANDLE_ERROR(cudaFree(d_blockStartRows));
        d_blockStartRows = nullptr;
    }

    if (timings.measureAll)
    {
        timings.loadBalanceNumeric = recordTimerVar(config.individualStart, config.individualEnd, stream);
        startTimerVar(config.individualStart, stream);
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  ALLOCATE GLOBAL MAPS
    // -------------------------------------------------------------------------------------------------------------------------------------------

    // always disabled since we always use dense mode for large rows
    supportGlobalFallback = false;
    if (supportGlobalFallback)
    {
        // update elements per map now that we know the lengths of each row --> could save some global memory and therefore allocation time
        elementsPerMap = max(maxElementsPerRow, maxNnzPerBlockNumericDynamicSharedMem) * 3 / 2;
        supportGlobalFallback &= h_blockCounterScaleNumeric[0] > 0;
        hashMapCount = min(sm * BLOCKS_PER_SM, h_blockCounterScaleNumeric[0]);
        hashMapCount = min(hashMapCount, rowsRequiringGlobal);
        supportGlobalFallback &= hashMapCount > 0;
    }

    rowsRequiringGlobal = matB.cols < entriesPerWarpNumeric * warpsNumeric ? 0 : rowsRequiringGlobal;
    bool isDenseOutput = h_blockCounterScaleNumeric[0] > 0;

    GlobalMapRowOffsets *rowOffsetMaps = nullptr;
    IndexType *rowOffsetMapIndices = nullptr;
    uint32_t rowOffsetMapCount = 0;
    uint32_t rowOffsetMapElementsPer = 0;

    if (isDenseOutput)
    {
        if (longestRowALength == 0)
        {
            uint32_t *d_longestRowALength = nullptr;
            HANDLE_ERROR(cudaMalloc(&d_longestRowALength, sizeof(uint32_t)));
            HANDLE_ERROR(cudaMemset(d_longestRowALength, 0, sizeof(uint32_t)));

            const uint32_t _threads = 256;
            const uint32_t rowsPerThread = 2;
            const uint32_t blocks = divup(IndexType(matA.rows), _threads * rowsPerThread);
            getLongestRowA<IndexType, _threads, rowsPerThread><<<blocks, _threads>>>(matA.row_offsets, d_longestRowALength, matA.rows, matA.nnz);

            cudaMemcpy(&longestRowALength, d_longestRowALength, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        }

        rowOffsetMapElementsPer = longestRowALength;
        rowOffsetMapCount = min(h_blockCounterScaleNumeric[0], sm * BLOCKS_PER_SM);

        // only allocate global maps if row cursors can't be held in share memory
        if (elementsPerMap * 2 * sizeof(IndexType) > warpsNumeric * entriesPerWarpNumeric * (sizeof(IndexType) + sizeof(DataType)))
        {
            if (h_blockCounterScaleNumeric[0] != 0)
            {
                if (rowOffsetMaps != nullptr)
                    HANDLE_ERROR(cudaFree(rowOffsetMaps));
                HANDLE_ERROR(cudaMalloc(&rowOffsetMaps, globalMapMaxSize * rowOffsetMapCount));

                if (rowOffsetMapIndices != nullptr)
                {
                    HANDLE_ERROR(cudaFree(rowOffsetMapIndices));
                    rowOffsetMapIndices = nullptr;
                }

                if (rowOffsetMapIndices == nullptr)
                    HANDLE_ERROR(cudaMalloc(&rowOffsetMapIndices, sizeof(IndexType) * rowOffsetMapCount * (rowOffsetMapElementsPer + maxRowsPerBlock + 1)));

                spgemm.setLaunchDimensions(rowOffsetMapCount, stream, 32 * warpsNumeric);
                spgemm.h_InitializeGlobalMapsNoVal<GlobalMapRowOffsets, uint32_t>((GlobalMapRowOffsets *)rowOffsetMaps, rowOffsetMapCount, rowOffsetMapIndices, rowOffsetMapElementsPer, maxRowsPerBlock);
            }
        }
    }

    if (timings.measureAll)
    {
        timings.globalMapsNumeric = recordTimerVar(config.individualStart, config.individualEnd, stream);
        startTimerVar(config.individualStart, stream);
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  PRE-NUMERIC LOAD OPTIMIZATIONS
    // -------------------------------------------------------------------------------------------------------------------------------------------

    // alloc indices for rows which shall be sorted by cub
    bool sortAllInplace = false;
    {
        {

            uint32_t activeSM = h_blockCounterScaleNumeric[0];
            // never go up to top level
            int firstXEmpty = 0;
            bool foundFirstNonEmpty = h_blockCounterScaleNumeric[0] != 0;
            for (int i = 1; i < kernelCountNumeric; ++i)
            {
                blockPrefixScaled[i] = h_blockCounterScaleNumeric[i - 1] + blockPrefixScaled[i - 1];
                activeSM += 2 * h_blockCounterScaleNumeric[i] >> (i - 1);
                if (!foundFirstNonEmpty)
                {
                    if (h_blockCounterScaleNumeric[i] == 0)
                        firstXEmpty++;
                    else
                        foundFirstNonEmpty = true;
                }
            }

            // avoid div by zero
            activeSM = max(activeSM, 1);

            if (activeSM < sm * BLOCKS_PER_SM)
            {
                int shiftUp = min(firstXEmpty, int(std::log2(sm * BLOCKS_PER_SM / activeSM)));

                if (shiftUp > 0)
                {
                    if (firstXEmpty >= 2)
                        sortAllInplace = true;

                    for (int i = 0; i < kernelCountNumeric; i++)
                    {
                        if (i + shiftUp < kernelCountNumeric)
                        {
                            h_blockCounterScaleNumeric[i] = h_blockCounterScaleNumeric[i + shiftUp];
                            blockPrefixScaled[i] = blockPrefixScaled[i + shiftUp];
                        }
                        else
                        {
                            h_blockCounterScaleNumeric[i] = 0;
                            blockPrefixScaled[i] = h_blockCounter;
                        }
                    }
                }
            }
        }

        // inplace starts to be faster if the size of the maps is getting smaller
		Config::SortModes sortMode = Config::SortModes::CubSegmentedSort;

        const uint32_t entrySize = sizeof(IndexType) + sizeof(DataType);

        Config::SpGEMMMethods spGemmMethodNumeric = Config::AutoSpGEMM;


        // -------------------------------------------------------------------------------------------------------------------------------------------
        //  NUMERIC SPGEMM
        // -------------------------------------------------------------------------------------------------------------------------------------------

        if (h_blockCounterScaleNumeric[0] > 0)
        {
            spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[0], streams[0], 32 * warpsNumeric, dynamicSharedBytesPerBlockNumeric);
            spgemm.h_DenseSpGEMMNumeric<IndexType, DataType, GlobalMapRowOffsets, dynamicSharedBytesPerBlockNumeric, true, (32 * warpsNumeric)>(
                matA, matB, matC, (GlobalMapRowOffsets *)rowOffsetMaps, rowOffsetMapCount,
                d_blockStartRows, d_rowOperations, h_blockCounterScaleNumeric[0], d_rowColMinMax,
                d_rowMaxOperations, false, rowsPerBlock);
        }

        sortMode = sortAllInplace ? Config::InPlace : Config::Separate;

        bool setSortingBit = sortAllInplace ? false : maxElementsPerRow >= 500;

        if (kernelCountNumeric > 1 && h_blockCounterScaleNumeric[1] > 0)
        {
            if (spGemmMethodNumeric == Config::AutoSpGEMM)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[1], streams[1], (32 * warpsNumeric >> 0), (sharedBytesPerBlockNumeric >> 0));
                spgemm.h_SpGEMMNumericLauncher<IndexType, DataType, GlobalMap, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 0), false, (32 * warpsNumeric >> 0)>(
                    matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount, rowOffsetMaps, rowOffsetMapCount,
                    d_blockStartRows + blockPrefixScaled[1], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[1], d_rowColMinMax, 
                    d_rowMaxOperations, denseModeRowThresholdExternalSorting, setSortingBit, rowsPerBlock);
            }
            else if (spGemmMethodNumeric == Config::DenseSpGEMM)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[1], streams[1], 32 * warpsNumeric >> 0, (sharedBytesPerBlockNumeric >> 0));
                spgemm.h_DenseSpGEMMNumeric<IndexType, DataType, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 0), false, (32 * warpsNumeric >> 0)>(
                    matA, matB, matC, (GlobalMapRowOffsets *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[1], d_rowOperations,
                    h_blockCounterScaleNumeric[1], d_rowColMinMax,
                    d_rowMaxOperations, setSortingBit, rowsPerBlock);
            }
            else
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[1], streams[1], 32 * warpsNumeric, (sharedBytesPerBlockNumeric >> 0));
                spgemm.h_HashSpGEMMNumeric<IndexType, DataType, GlobalMap, (sharedBytesPerBlockNumeric >> 0), false, (32 * warpsNumeric)>(
                    matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[1], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[1], d_rowColMinMax,
                    d_rowMaxOperations, setSortingBit, rowsPerBlock);
            }
        }

        if (kernelCountNumeric > 2 && h_blockCounterScaleNumeric[2] > 0)
        {
            if (spGemmMethodNumeric == Config::AutoSpGEMM)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[2], streams[2], (32 * warpsNumeric >> 1), (sharedBytesPerBlockNumeric >> 1));
                spgemm.h_SpGEMMNumericLauncher<IndexType, DataType, GlobalMap, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 1), false, (32 * warpsNumeric >> 1)>(
                    matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount, rowOffsetMaps, rowOffsetMapCount,
                    d_blockStartRows + blockPrefixScaled[2], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[2], d_rowColMinMax,
                    d_rowMaxOperations, denseModeRowThresholdExternalSorting, setSortingBit, rowsPerBlock);
            }
            else if (spGemmMethodNumeric == Config::DenseSpGEMM)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[2], streams[2], 32 * warpsNumeric >> 1, (sharedBytesPerBlockNumeric >> 1));
                spgemm.h_DenseSpGEMMNumeric<IndexType, DataType, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 1), false, (32 * warpsNumeric >> 1)>(
                    matA, matB, matC, (GlobalMapRowOffsets *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[2], d_rowOperations,
                    h_blockCounterScaleNumeric[2], d_rowColMinMax,
                    d_rowMaxOperations, setSortingBit, rowsPerBlock);
            }
            else
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[2], streams[2], 32 * warpsNumeric >> 1, (sharedBytesPerBlockNumeric >> 1));
                spgemm.h_HashSpGEMMNumeric<IndexType, DataType, GlobalMap, (sharedBytesPerBlockNumeric >> 1), false, (32 * warpsNumeric >> 1)>(
                    matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[2], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[2], d_rowColMinMax,
                    d_rowMaxOperations, setSortingBit, rowsPerBlock);
            }
        }

        sortMode = Config::InPlace;

        if (kernelCountNumeric > 3 && h_blockCounterScaleNumeric[3] > 0)
        {
            if (spGemmMethodNumeric == Config::AutoSpGEMM)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[3], streams[3], (32 * warpsNumeric >> 2), (sharedBytesPerBlockNumeric >> 2));
                spgemm.h_SpGEMMNumericLauncher<IndexType, DataType, GlobalMap, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 2), false, (32 * warpsNumeric >> 2)>(
                    matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount, rowOffsetMaps, rowOffsetMapCount,
                    d_blockStartRows + blockPrefixScaled[3], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[3], d_rowColMinMax,
                    d_rowMaxOperations, denseModeRowThresholdInternalSorting, false, rowsPerBlock);
            }
            else if (spGemmMethodNumeric == Config::DenseSpGEMM)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[3], streams[3], 32 * warpsNumeric >> 2, (sharedBytesPerBlockNumeric >> 2));
                spgemm.h_DenseSpGEMMNumeric<IndexType, DataType, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 2), false, (32 * warpsNumeric >> 2)>(
                    matA, matB, matC, (GlobalMapRowOffsets *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[3], d_rowOperations,
                    h_blockCounterScaleNumeric[3], d_rowColMinMax,
                    d_rowMaxOperations, false, rowsPerBlock);
            }
            else
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[3], streams[3], 32 * warpsNumeric >> 2, (sharedBytesPerBlockNumeric >> 2));
                spgemm.h_HashSpGEMMNumeric<IndexType, DataType, GlobalMap, (sharedBytesPerBlockNumeric >> 2), false, (32 * warpsNumeric >> 2)>(
                    matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[3], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[3], d_rowColMinMax,
                    d_rowMaxOperations, false, rowsPerBlock);
            }
        }

        if (kernelCountNumeric > 4 && h_blockCounterScaleNumeric[4] > 0)
        {
            if (spGemmMethodNumeric == Config::AutoSpGEMM)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[4], streams[4], (32 * warpsNumeric >> 3), (sharedBytesPerBlockNumeric >> 3));
                spgemm.h_SpGEMMNumericLauncher<IndexType, DataType, GlobalMap, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 3), false, (32 * warpsNumeric >> 3)>(
                    matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount, rowOffsetMaps, rowOffsetMapCount,
                    d_blockStartRows + blockPrefixScaled[4], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[4], d_rowColMinMax,
                    d_rowMaxOperations, denseModeRowThresholdInternalSorting, false, rowsPerBlock);
            }
            else if (spGemmMethodNumeric == Config::DenseSpGEMM)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[4], streams[4], 32 * warpsNumeric >> 3, (sharedBytesPerBlockNumeric >> 3));
                spgemm.h_DenseSpGEMMNumeric<IndexType, DataType, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 3), false, (32 * warpsNumeric >> 3)>(
                    matA, matB, matC, (GlobalMapRowOffsets *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[4], d_rowOperations,
                    h_blockCounterScaleNumeric[4], d_rowColMinMax,
                    d_rowMaxOperations, false, rowsPerBlock);
            }
            else
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[4], streams[4], 32 * warpsNumeric >> 3, (sharedBytesPerBlockNumeric >> 3));
                spgemm.h_HashSpGEMMNumeric<IndexType, DataType, GlobalMap, (sharedBytesPerBlockNumeric >> 3), false, (32 * warpsNumeric >> 3)>(
                    matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[4], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[4], d_rowColMinMax,
                    d_rowMaxOperations, false, rowsPerBlock);
            }
        }

        if (kernelCountNumeric > 5 && h_blockCounterScaleNumeric[5] > 0)
        {
            if (spGemmMethodNumeric == Config::AutoSpGEMM || ((rowsPerBlock > 1 || reprocessLoadBalanceNumeric) && spGemmMethodNumeric != Config::HashSpGEMM))
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[5], streams[5], (32 * warpsNumeric >> 4), (sharedBytesPerBlockNumeric >> 4));
                spgemm.h_SpGEMMNumericLauncher<IndexType, DataType, GlobalMap, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 4), false, (32 * warpsNumeric >> 4)>(
                    matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount, rowOffsetMaps, rowOffsetMapCount,
                    d_blockStartRows + blockPrefixScaled[5], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[5], d_rowColMinMax,
                    d_rowMaxOperations, denseModeRowThresholdInternalSorting, false, rowsPerBlock);
            }
            else if (spGemmMethodNumeric == Config::DenseSpGEMM)
            {

                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[5], streams[5], 32 * warpsNumeric >> 4, (sharedBytesPerBlockNumeric >> 4));
                spgemm.h_DenseSpGEMMNumeric<IndexType, DataType, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 4), false, (32 * warpsNumeric >> 4)>(
                    matA, matB, matC, (GlobalMapRowOffsets *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[5], d_rowOperations,
                    h_blockCounterScaleNumeric[5], d_rowColMinMax,
                    d_rowMaxOperations, false, rowsPerBlock);
            }
            else
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[5], streams[5], 32 * warpsNumeric >> 4, (sharedBytesPerBlockNumeric >> 4));
                spgemm.h_HashSpGEMMNumeric<IndexType, DataType, GlobalMap, (sharedBytesPerBlockNumeric >> 4), false, (32 * warpsNumeric >> 4)>(
                    matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[5], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[5], d_rowColMinMax,
                    d_rowMaxOperations, false, rowsPerBlock);
            }
        }
    }

    if (timings.measureAll)
    {
        HANDLE_ERROR(cudaDeviceSynchronize());
        timings.spGEMMNumeric = recordTimerVar(config.individualStart, config.individualEnd, stream);
        startTimerVar(config.individualStart, stream);
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  SORT MEDIUM AND LONG ROWS
    // -------------------------------------------------------------------------------------------------------------------------------------------

    if (!sortAllInplace && (h_blockCounterScaleNumeric[1] + h_blockCounterScaleNumeric[2] > 0) && maxElementsPerRow >= 500)
    {
        if (h_blockCounterScaleNumeric[2] > 0)
        {
            spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[2], streams[2], 32 * warpsNumeric / 4);
            spgemm.h_HashSpGEMMSorting<uint32_t, DataType, 32 * warpsNumeric / 4, entriesPerWarpNumeric * 32 / 2>(
                matC, d_blockStartRows + blockPrefixScaled[2], h_blockCounterScaleNumeric[2], true);
        }

        if (h_blockCounterScaleNumeric[1] > 0)
        {
            spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[1], streams[1], 32 * warpsNumeric / 2);
            spgemm.h_HashSpGEMMSorting<uint32_t, DataType, 32 * warpsNumeric / 2, entriesPerWarpNumeric * 32>(
                matC, d_blockStartRows + blockPrefixScaled[1], h_blockCounterScaleNumeric[1], true);
        }
    }

    if (timings.measureAll)
    {
        HANDLE_ERROR(cudaDeviceSynchronize());
        timings.sorting = recordTimerVar(config.individualStart, config.individualEnd, stream);
        startTimerVar(config.individualStart, stream);
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  FREE ALLOCATED MEMORY
    // -------------------------------------------------------------------------------------------------------------------------------------------

    if (d_blockStartRows != nullptr)
        HANDLE_ERROR(cudaFree(d_blockStartRows));
    if (hashMaps != nullptr)
        HANDLE_ERROR(cudaFree(hashMaps));
    if (maps_indices != nullptr)
        HANDLE_ERROR(cudaFree(maps_indices));
    if (maps_values != nullptr)
        HANDLE_ERROR(cudaFree(maps_values));

    if (d_combined_pointers != nullptr)
        HANDLE_ERROR(cudaFree(d_combined_pointers));

    if (rowOffsetMaps != nullptr)
        HANDLE_ERROR(cudaFree(rowOffsetMaps));
    if (rowOffsetMapIndices != nullptr)
        HANDLE_ERROR(cudaFree(rowOffsetMapIndices));

    if (timings.measureAll)
    {
        timings.cleanup = recordTimerVar(config.individualStart, config.individualEnd, stream);
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  END
    // -------------------------------------------------------------------------------------------------------------------------------------------

    if (timings.measureCompleteTime) {
        HANDLE_ERROR(cudaDeviceSynchronize());
        timings.complete = recordTimerVar(config.completeStart, config.completeEnd, stream);
    }

    if (timings.measureAll)
    {
		/*printf("elements per global map=%d. mapCount=%d\n", elementsPerMap, hashMapCount);
        printf("matCNnz=%d, number of blocks = %d, %d, %d, %d, %d, %d\n", matC.nnz,
            h_blockCounterScaleNumeric[0],
            kernelCountNumeric > 1 ? h_blockCounterScaleNumeric[1] : -1,
            kernelCountNumeric > 2 ? h_blockCounterScaleNumeric[2] : -1,
            kernelCountNumeric > 3 ? h_blockCounterScaleNumeric[3] : -1,
            kernelCountNumeric > 4 ? h_blockCounterScaleNumeric[4] : -1,
            kernelCountNumeric > 5 ? h_blockCounterScaleNumeric[5] : -1);*/
        if (timings.measureAll)
        {
            printf("spECK     initial mallocs = %f ms\n", timings.init);
            printf("spECK  count computations = %f ms\n", timings.countProducts);
            printf("spECK       load-balancer = %f ms\n", timings.loadBalanceCounting);
            printf("spECK      GlobalMaps Cnt = %f ms\n", timings.globalMapsCounting);
            printf("spECK     counting kernel = %f ms\n", timings.spGEMMCounting);
            printf("spECK        malloc mat C = %f ms\n", timings.allocC);
            printf("spECK   num load-balancer = %f ms\n", timings.loadBalanceNumeric);
            printf("spECK     init GlobalMaps = %f ms\n", timings.globalMapsNumeric);
            printf("spECK      numeric kernel = %f ms\n", timings.spGEMMNumeric);
            printf("spECK      Sorting kernel = %f ms\n", timings.sorting);
            printf("spECK             cleanup = %f ms\n", timings.cleanup);
            printf("--------------------------------------------------------------\n");
        }
        if (timings.measureCompleteTime)
            printf("spECK            complete = %f ms\n\n", timings.complete);
    }

    matOut.rows = matC.rows;
    matOut.cols = matC.cols;
    matOut.nnz = matC.nnz;
    matOut.col_ids = matC.col_ids;
    matOut.row_offsets = matC.row_offsets;
    matOut.data = matC.data;
}

template <typename DataType, int BLOCKS_PER_SM, int THREADS_PER_BLOCK, int MAX_DYNAMIC_SHARED, int MAX_STATIC_SHARED>
void MultiplyspECK(const dCSR<DataType> &A, const dCSR<DataType> &B, dCSR<DataType> &matOut, spECKConfig &config, Timings &timings)
{
	MultiplyspECKImplementation<DataType, BLOCKS_PER_SM, THREADS_PER_BLOCK, MAX_DYNAMIC_SHARED, MAX_STATIC_SHARED>(A, B, matOut, config, timings);
}

template void MultiplyspECK<float, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>(const dCSR<float> &A, const dCSR<float> &B, dCSR<float> &matOut, spECKConfig &config, Timings &timings);
template void MultiplyspECK<double, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>(const dCSR<double> &A, const dCSR<double> &B, dCSR<double> &matOut, spECKConfig &config, Timings &timings);
// template void MultiplyspECK<uint64_t, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>(const dCSR<uint64_t> &A, const dCSR<uint64_t> &B, dCSR<uint64_t> &matOut, spECKConfig &config, Timings &timings);
} // namespace spECK
