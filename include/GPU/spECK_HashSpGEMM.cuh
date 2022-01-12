#pragma once

#include "dCSR.h"
#include "common.cuh"
#include "HashMap.cuh"
#include <device_launch_parameters.h>
#include "GPU/spECKKernels.h"
#include "Config.h"

// saves all entries from the shared memory map to global map in counting kernel
template <class GlobalMap, class LocalMap>
__device__ __forceinline__ void saveSMemMapToGlobalNoVal(LocalMap *sMemMap, GlobalMap *globalMap)
{
	for (int i = threadIdx.x; i < sMemMap->getSize(); i += blockDim.x)
	{
		auto id = sMemMap->ids[i];
		if (id != sMemMap->UNUSED())
		{
			globalMap->at(sMemMap->idToRow(id), sMemMap->idToCol(id));
		}
	}
}

// saves all entries from the shared memory map to global map in numeric kernel
template <uint32_t MAP_SIZE, uint32_t THREADS, class GlobalMap, class LocalMap>
__device__ __forceinline__ void saveSMemMapToGlobal(LocalMap *sMemMap, GlobalMap *globalMap)
{
	for (int i = threadIdx.x; i < MAP_SIZE; i += THREADS)
	{
		auto id = sMemMap->ids[i];
		if (id != sMemMap->UNUSED())
		{
			globalMap->atomic_add(sMemMap->idToRow(id), sMemMap->idToCol(id), sMemMap->values[i]);
		}
	}
}

// the actual loop of the hash based counting kernel
template <typename INDEX_TYPE, uint32_t MAP_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS, bool DIRECT, uint32_t SHIFT, class GlobalMap, class LocalMap>
__device__ void iterateMatrixCounting(INDEX_TYPE startId, INDEX_TYPE lastIdExcl, INDEX_TYPE startRow, INDEX_TYPE lastRowExcl,
	const INDEX_TYPE* __restrict__ rowOffsetsA, const INDEX_TYPE* __restrict__ rowOffsetsB,
	const INDEX_TYPE* __restrict__ colIdsA, const INDEX_TYPE* __restrict__ colIdsB,
	const INDEX_TYPE nnzA, const INDEX_TYPE nnzB, const INDEX_TYPE rowsA, const INDEX_TYPE rowsB, 
	bool supportGlobal,	GlobalMap** gMap, LocalMap& lMap,
	GlobalMap* __restrict__  maps, INDEX_TYPE mapCount, INDEX_TYPE resultNnz, INDEX_TYPE minCol)
{
	// if not all threads can write value into map anymore, we have to switch to global map
	const uint32_t maxOccupancy = MAP_SIZE - THREADS;
	bool switchToGlobal = false;
	// in case of 'restart', lastIdB holds the last position
	INDEX_TYPE lastIdB = spECK::numeric_limits<INDEX_TYPE>::max();

	// every thread stays for the exact same amount of iterations in the loop
	for (INDEX_TYPE _idA = startId; _idA < lastIdExcl; _idA += THREADS >> SHIFT)
	{
		// this thread looks at all non zero cols of B
		INDEX_TYPE idA = (threadIdx.x >> SHIFT) + _idA;
		uint32_t valid = idA < lastIdExcl ? 1U : 0U;

		// find row which contains idA
		INDEX_TYPE rowA = startRow;
		if (lastRowExcl - startRow > 1) {
			for (; valid && rowA < lastRowExcl; ++rowA)
			{
				if (rowOffsetsA[rowA] <= idA && (rowA + 1 < rowsA ? rowOffsetsA[rowA + 1] : nnzA) > idA)
					break;
			} 
		}

		INDEX_TYPE colA = valid ? colIdsA[idA] : 0;
		INDEX_TYPE rowStartB = valid ? rowOffsetsB[colA] : spECK::numeric_limits<INDEX_TYPE>::max();
		INDEX_TYPE rowEndB = valid ? (colA + 1 < rowsB ? rowOffsetsB[colA + 1] : nnzB) : 0;

		// tId % THREADS_PER_ROW
		rowStartB += threadIdx.x & ((1 << SHIFT) - 1);

		if (SUPPORT_GLOBAL && supportGlobal && switchToGlobal)
		{
			// block just flushed shared memory map, continue where we left off

			rowStartB = lastIdB;
			// make sure that a thread which could finish will not add any new items this time
			lastIdB = spECK::numeric_limits<INDEX_TYPE>::max();
			switchToGlobal = false;
		}

		for (INDEX_TYPE idB = rowStartB; idB < rowEndB; idB += (1 << SHIFT))
		{
			if (SUPPORT_GLOBAL && supportGlobal && *lMap.occupancy >= maxOccupancy)
			{
				switchToGlobal = true;
				lastIdB = idB;
				break;
			}

			auto colB = colIdsB[idB];

			// skip hashing and use direct array access if the range of columns in C is smaller than hash map size
			if (DIRECT)
				lMap.atDirect(rowA - startRow, colB - minCol);
			else
				lMap.at(rowA - startRow, colB);
		}

		// not all threads can add a new entry to the shared memory map -> flush it into global and continue
		if (SUPPORT_GLOBAL && supportGlobal && __syncthreads_or(switchToGlobal))
		{
			// shared memory map is full -> allocate free map in global and write out all values
			if (threadIdx.x == 0 && *gMap == nullptr)
			{
				*gMap = reserveMap<GlobalMap>(maps, mapCount);
				(*gMap)->limitSize(resultNnz * 2);
			}

			__syncthreads();
			saveSMemMapToGlobalNoVal(&lMap, *gMap);
			__syncthreads();
		
			// reset shared memory map and continue writing into shared memory map
			lMap.init(threadIdx.x == 0);
			__syncthreads();

			// subtract row iteration, because for loop will increment it
			_idA -= THREADS >> SHIFT;
			switchToGlobal = true;
		}
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE, uint32_t MAP_SIZE, uint32_t THREADS, bool DIRECT, uint32_t SHIFT, class LocalMap>
__device__ void iterateMatrixNumeric(INDEX_TYPE startId, INDEX_TYPE lastIdExcl, INDEX_TYPE startRow, INDEX_TYPE lastRowExcl,
	const INDEX_TYPE* __restrict__ rowOffsetsA, const INDEX_TYPE* __restrict__ rowOffsetsB,
	const INDEX_TYPE* __restrict__ colIdsA, const INDEX_TYPE* __restrict__ colIdsB,
	const VALUE_TYPE* __restrict__ valuesA, const VALUE_TYPE* __restrict__ valuesB,
	const INDEX_TYPE nnzA, const INDEX_TYPE nnzB, const INDEX_TYPE rowsA, const INDEX_TYPE rowsB,
	LocalMap& __restrict__ lMap,
	INDEX_TYPE resultNnz, INDEX_TYPE minCol)
{
	// every thread stays for the exact same amount of iterations in the loop
	for (INDEX_TYPE idA = startId + (threadIdx.x >> SHIFT); idA < lastIdExcl; idA += THREADS >> SHIFT)
	{
		// find row which contains idA
		INDEX_TYPE rowA = startRow;
		for (;rowA < lastRowExcl; ++rowA)
		{
			if (rowOffsetsA[rowA] <= idA && rowOffsetsA[rowA + 1] > idA)
				break;
		}

		INDEX_TYPE colA = colIdsA[idA];
		// tId % THREADS_PER_ROW
		INDEX_TYPE rowStartB = rowOffsetsB[colA] + (threadIdx.x & ((1 << SHIFT) - 1));
		INDEX_TYPE rowEndB = rowOffsetsB[colA + 1];

		for (INDEX_TYPE idB = rowStartB; idB < rowEndB; idB += (1 << SHIFT))
		{
			auto colB = colIdsB[idB];
			auto valA = valuesA[idA];
			auto valB = valuesB[idB];

			// skip hashing and use direct array access if the range of columns in C is smaller than hash map size
			if (DIRECT)
				lMap.atomic_add_direct(rowA - startRow, colB - minCol, valA * valB);
			else
				lMap.atomic_add(rowA - startRow, colB, valA * valB);
		}
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE, uint32_t THREADS, bool DIRECT, uint32_t SHIFT, class LocalMap>
__device__ void iterateMatrixNumericSingleRow(
	const INDEX_TYPE startId, const INDEX_TYPE lastIdExcl,
	const INDEX_TYPE* __restrict__ rowOffsetsB,
	const INDEX_TYPE* __restrict__ colIdsA, const INDEX_TYPE* __restrict__ colIdsB,
	const VALUE_TYPE* __restrict__ valuesA, const VALUE_TYPE* __restrict__ valuesB,
	const INDEX_TYPE nnzB, const INDEX_TYPE rowsB, 
	const LocalMap& lMap, INDEX_TYPE resultNnz, INDEX_TYPE minCol)
{
	// every thread stays for the exact same amount of iterations in the loop
	for (INDEX_TYPE idA = startId + (threadIdx.x >> SHIFT); idA < lastIdExcl; idA += THREADS >> SHIFT)
	{
		INDEX_TYPE colA = colIdsA[idA];
		INDEX_TYPE rowStartB = rowOffsetsB[colA] + (threadIdx.x & ((1 << SHIFT) - 1));
		INDEX_TYPE rowEndB = rowOffsetsB[colA + 1];
		auto valA = valuesA[idA];

		for (INDEX_TYPE idB = rowStartB; idB < rowEndB; idB += (1 << SHIFT))
		{
			auto colB = colIdsB[idB];
			auto valB = valuesB[idB];

			// skip hashing and use direct array access if the range of columns in C is smaller than hash map size
			if (DIRECT) {					
				atomicAdd(&lMap.values[colB - minCol], valA * valB);
				lMap.ids[colB - minCol] = colB - minCol;
			}
			else {

				INDEX_TYPE hashed_id = currentHash(colB);
				INDEX_TYPE map_id = hashed_id % lMap.getSize();
				do
				{
					auto old_id = atomicCAS(lMap.ids + map_id, 0xFFFFFFFFU, colB);
					if (old_id == 0xFFFFFFFFU || old_id == colB)
					{
						atomicAdd(lMap.values + map_id, valA * valB);
						break;
					}

					map_id = (map_id + 1) % lMap.getSize();
				} while (true);
			}
		}
	}
}
template <typename INDEX_TYPE, uint32_t THREADS, bool DIRECT, uint32_t SHIFT, class LocalMap>
__device__ void iterateMatrixCountingSingleRow(
	const INDEX_TYPE startId, const INDEX_TYPE lastIdExcl,
	const INDEX_TYPE* __restrict__ rowOffsetsB,
	const INDEX_TYPE* __restrict__ colIdsA, const INDEX_TYPE* __restrict__ colIdsB,
	const LocalMap& lMap, INDEX_TYPE minCol)
{

	// every thread stays for the exact same amount of iterations in the loop
	for (INDEX_TYPE idA = startId + (threadIdx.x >> SHIFT); idA < lastIdExcl; idA += THREADS >> SHIFT)
	{
		INDEX_TYPE colA = colIdsA[idA];
		INDEX_TYPE rowStartB = rowOffsetsB[colA] + (threadIdx.x & ((1 << SHIFT) - 1));
		INDEX_TYPE rowEndB = rowOffsetsB[colA + 1];

		for (INDEX_TYPE idB = rowStartB; idB < rowEndB; idB += (1 << SHIFT))
		{
			auto colB = colIdsB[idB];

			// skip hashing and use direct array access if the range of columns in C is smaller than hash map size
			if (DIRECT) {
				auto old_id = lMap.ids[colB - minCol];
				if (old_id == 0xFFFFFFFFU && atomicCAS(lMap.ids + colB - minCol, 0xFFFFFFFFU, colB) == 0xFFFFFFFFU)
				{
					atomicAdd(lMap.occupancyPerRow, 1);
				}
			}
			else {
				INDEX_TYPE hashed_id = currentHash(colB);
				INDEX_TYPE map_id = hashed_id % lMap.getSize();
				do
				{
					auto old_id = lMap.ids[map_id];
					if (old_id == colB)
					{
						break;
					}
					if (old_id == 0xFFFFFFFFU)
					{
						old_id = atomicCAS(lMap.ids + map_id, 0xFFFFFFFFU, colB);
						if (old_id == 0xFFFFFFFFU)
						{
							atomicAdd(lMap.occupancyPerRow, 1);
							break;
						}
						else if (old_id == colB)
						{
							break;
						}
					}

					map_id = (map_id + 1) % lMap.getSize();
				} while (true);
			}
		}
	}
}

// this class creates all instances of my iterateMatrix template and selects the best instance for the specific call 'call'
template <typename INDEX_TYPE, uint32_t MAP_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS, class GlobalMap, class LocalMap>
class IterateMatrixCountMethods
{
#define iterateCount(startId, lastIdExcl, shift, global, direct) call_template<global, direct>( \
	startId, lastIdExcl, shift, startRow, lastRowExcl, \
	rowOffsetsA, rowOffsetsB, colIdsA, colIdsB, \
	nnzA, nnzB, rowsA, rowsB,\
	gMap, lMap, maps, mapCount, resultNnz, minCol);

#define iterate_templ_Count(startId, lastIdExcl, shift, global, direct) iterateMatrixCounting<INDEX_TYPE, MAP_SIZE, global, THREADS, direct, shift>( \
	startId, lastIdExcl, startRow, lastRowExcl, \
	rowOffsetsA, rowOffsetsB, colIdsA, colIdsB, \
	nnzA, nnzB, rowsA, rowsB,\
	global, gMap, lMap, maps, mapCount, resultNnz, minCol);

private:
	template<bool GLOBAL, bool DIRECT>
	__device__ static __forceinline__ void call_template(INDEX_TYPE startId, INDEX_TYPE lastIdExcl, uint32_t shift, INDEX_TYPE startRow, INDEX_TYPE lastRowExcl,
		const INDEX_TYPE* __restrict__ rowOffsetsA, const INDEX_TYPE* __restrict__ rowOffsetsB,
		const INDEX_TYPE* __restrict__ colIdsA, const INDEX_TYPE* __restrict__ colIdsB,
		const INDEX_TYPE nnzA, const INDEX_TYPE nnzB, const INDEX_TYPE rowsA, const INDEX_TYPE rowsB,
		GlobalMap** gMap, LocalMap& __restrict__ lMap,
		GlobalMap* __restrict__  maps, INDEX_TYPE mapCount, INDEX_TYPE resultNnz, INDEX_TYPE minCol)
	{
		if (shift == 0)
			iterate_templ_Count(startId, lastIdExcl, 0, GLOBAL, DIRECT)
		else if (shift == 1)
			iterate_templ_Count(startId, lastIdExcl, 1, GLOBAL, DIRECT)
		else if (shift == 2)
			iterate_templ_Count(startId, lastIdExcl, 2, GLOBAL, DIRECT)
		else if (shift == 3)
			iterate_templ_Count(startId, lastIdExcl, 3, GLOBAL, DIRECT)
		else if (shift == 4)
			iterate_templ_Count(startId, lastIdExcl, 4, GLOBAL, DIRECT)
		else if (shift == 5)
			iterate_templ_Count(startId, lastIdExcl, 5, GLOBAL, DIRECT)
		else if (shift == 6)
			iterate_templ_Count(startId, lastIdExcl, 6, GLOBAL, DIRECT)
		else if (shift == 7)
			iterate_templ_Count(startId, lastIdExcl, 7, GLOBAL, DIRECT)
		else if (shift == 8)
			iterate_templ_Count(startId, lastIdExcl, 8, GLOBAL, DIRECT)
		else if (shift == 9)
			iterate_templ_Count(startId, lastIdExcl, 9, GLOBAL, DIRECT)
		else if (shift == 10)
			iterate_templ_Count(startId, lastIdExcl, 10, GLOBAL, DIRECT)
	}

public:
	__device__ static __forceinline__ void call(INDEX_TYPE startId, INDEX_TYPE lastIdExcl, uint32_t shift, bool direct, INDEX_TYPE startRow, INDEX_TYPE lastRowExcl,
		const INDEX_TYPE* __restrict__ rowOffsetsA, const INDEX_TYPE* __restrict__ rowOffsetsB,
		const INDEX_TYPE* __restrict__ colIdsA, const INDEX_TYPE* __restrict__ colIdsB,
		const INDEX_TYPE nnzA, const INDEX_TYPE nnzB, const INDEX_TYPE rowsA, const INDEX_TYPE rowsB, 
		bool supportGlobal,	GlobalMap** gMap, LocalMap& __restrict__ lMap,
		GlobalMap* __restrict__  maps, INDEX_TYPE mapCount, INDEX_TYPE resultNnz, INDEX_TYPE minCol)
	{
		if (direct == true)
		{
			if (supportGlobal)
				iterateCount(startId, lastIdExcl, shift, true, true)
			else
				iterateCount(startId, lastIdExcl, shift, false, true)
		}
		else
		{
			if (supportGlobal)
				iterateCount(startId, lastIdExcl, shift, true, false)
			else
				iterateCount(startId, lastIdExcl, shift, false, false)
		}
	}
};

// this class creates all instances of my iterateMatrix template and selects the best instance for the specific call 'call'
template <typename INDEX_TYPE, typename VALUE_TYPE, uint32_t MAP_SIZE, uint32_t THREADS, class LocalMap>
class IterateMatrixNumericMethods
{
#define iterateNumeric(startId, lastIdExcl, shift, direct) call_template<direct>( \
	startId, lastIdExcl, shift, startRow, lastRowExcl, \
	rowOffsetsA, rowOffsetsB, colIdsA, colIdsB, valuesA, valuesB, \
	nnzA, nnzB, rowsA, rowsB,\
	lMap, resultNnz, minCol);

#define iterate_templ_Numeric(startId, lastIdExcl, shift, direct) iterateMatrixNumeric<INDEX_TYPE, VALUE_TYPE, MAP_SIZE, THREADS, direct, shift>( \
	startId, lastIdExcl, startRow, lastRowExcl, \
	rowOffsetsA, rowOffsetsB, colIdsA, colIdsB, valuesA, valuesB, \
	nnzA, nnzB, rowsA, rowsB,\
	lMap, resultNnz, minCol);

private:
	template<bool DIRECT>
	__device__ static __forceinline__ void call_template(INDEX_TYPE startId, INDEX_TYPE lastIdExcl, uint32_t shift, INDEX_TYPE startRow, INDEX_TYPE lastRowExcl,
		const INDEX_TYPE* __restrict__ rowOffsetsA, const INDEX_TYPE* __restrict__ rowOffsetsB,
		const INDEX_TYPE* __restrict__ colIdsA, const INDEX_TYPE* __restrict__ colIdsB,
		const VALUE_TYPE* __restrict__ valuesA, const VALUE_TYPE* __restrict__ valuesB,
		const INDEX_TYPE nnzA, const INDEX_TYPE nnzB, const INDEX_TYPE rowsA, const INDEX_TYPE rowsB,
		LocalMap& __restrict__ lMap,
		INDEX_TYPE resultNnz, INDEX_TYPE minCol)
	{
		if (shift == 0)
			iterate_templ_Numeric(startId, lastIdExcl, 0, DIRECT)
		else if (shift == 1)
			iterate_templ_Numeric(startId, lastIdExcl, 1, DIRECT)
		else if (shift == 2)
			iterate_templ_Numeric(startId, lastIdExcl, 2, DIRECT)
		else if (shift == 3)
			iterate_templ_Numeric(startId, lastIdExcl, 3, DIRECT)
		else if (shift == 4)
			iterate_templ_Numeric(startId, lastIdExcl, 4, DIRECT)
		else if (shift == 5)
			iterate_templ_Numeric(startId, lastIdExcl, 5, DIRECT)
		else if (shift == 6)
			iterate_templ_Numeric(startId, lastIdExcl, 6, DIRECT)
		else if (shift == 7)
			iterate_templ_Numeric(startId, lastIdExcl, 7, DIRECT)
		else if (shift == 8)
			iterate_templ_Numeric(startId, lastIdExcl, 8, DIRECT)
		else if (shift == 9)
			iterate_templ_Numeric(startId, lastIdExcl, 9, DIRECT)
		else if (shift == 10)
			iterate_templ_Numeric(startId, lastIdExcl, 10, DIRECT)
	}

public:
	__device__ static __forceinline__ void call(INDEX_TYPE startId, INDEX_TYPE lastIdExcl, int shift, bool direct, INDEX_TYPE startRow, INDEX_TYPE lastRowExcl,
		const INDEX_TYPE* __restrict__ rowOffsetsA, const INDEX_TYPE* __restrict__ rowOffsetsB,
		const INDEX_TYPE* __restrict__ colIdsA, const INDEX_TYPE* __restrict__ colIdsB,
		const VALUE_TYPE* __restrict__ valuesA, const VALUE_TYPE* __restrict__ valuesB,
		const INDEX_TYPE nnzA, const INDEX_TYPE nnzB, const INDEX_TYPE rowsA, const INDEX_TYPE rowsB, 
		LocalMap& __restrict__ lMap,
		INDEX_TYPE resultNnz, INDEX_TYPE minCol)
	{
		if (direct == true)
			iterateNumeric(startId, lastIdExcl, shift, true)
		else
			iterateNumeric(startId, lastIdExcl, shift, false)
	}
};

// this class creates all instances of my iterateMatrix template and selects the best instance for the specific call 'call'
template <typename INDEX_TYPE, typename VALUE_TYPE, uint32_t THREADS, class LocalMap>
class IterateMatrixNumericSingleRowMethods
{
#define iterateNumeric(startId, lastIdExcl, shift, direct) call_template<direct>( \
	startId, lastIdExcl, shift,  \
	rowOffsetsB, colIdsA, colIdsB, valuesA, valuesB, \
	nnzB, rowsB,\
	lMap, resultNnz, minCol);

#define iterate_templ_Numeric(startId, lastIdExcl, shift, direct) iterateMatrixNumericSingleRow<INDEX_TYPE, VALUE_TYPE, THREADS, direct, shift>( \
	startId, lastIdExcl,                                                                                                                             \
	rowOffsetsB, colIdsA, colIdsB, valuesA, valuesB,                                                                                                          \
	nnzB, rowsB,                                                                                                                                              \
	lMap, resultNnz, minCol);

private:
	template<bool DIRECT>
	__device__ static __forceinline__ void call_template(
		INDEX_TYPE startId, INDEX_TYPE lastIdExcl, uint32_t shift,
		const INDEX_TYPE* __restrict__ rowOffsetsB,
		const INDEX_TYPE* __restrict__ colIdsA, const INDEX_TYPE* __restrict__ colIdsB,
		const VALUE_TYPE* __restrict__ valuesA, const VALUE_TYPE* __restrict__ valuesB,
		const INDEX_TYPE nnzB, const INDEX_TYPE rowsB,
		LocalMap& __restrict__ lMap,
		INDEX_TYPE resultNnz, INDEX_TYPE minCol)
	{
		if (shift == 0)
			iterate_templ_Numeric(startId, lastIdExcl, 0, DIRECT)
		else if (shift == 1)
			iterate_templ_Numeric(startId, lastIdExcl, 1, DIRECT)
		else if (shift == 2)
			iterate_templ_Numeric(startId, lastIdExcl, 2, DIRECT)
		else if (shift == 3)
			iterate_templ_Numeric(startId, lastIdExcl, 3, DIRECT)
		else if (shift == 4)
			iterate_templ_Numeric(startId, lastIdExcl, 4, DIRECT)
		else if (shift == 5)
			iterate_templ_Numeric(startId, lastIdExcl, 5, DIRECT)
		else if (shift == 6)
			iterate_templ_Numeric(startId, lastIdExcl, 6, DIRECT)
		else if (shift == 7)
			iterate_templ_Numeric(startId, lastIdExcl, 7, DIRECT)
		else if (shift == 8)
			iterate_templ_Numeric(startId, lastIdExcl, 8, DIRECT)
		else if (shift == 9)
			iterate_templ_Numeric(startId, lastIdExcl, 9, DIRECT)
		else if (shift == 10)
			iterate_templ_Numeric(startId, lastIdExcl, 10, DIRECT)
	}

public:
	__device__ static __forceinline__ void call(INDEX_TYPE startId, INDEX_TYPE lastIdExcl, int shift, bool direct,
		const INDEX_TYPE* __restrict__ rowOffsetsB,
		const INDEX_TYPE* __restrict__ colIdsA, const INDEX_TYPE* __restrict__ colIdsB,
		const VALUE_TYPE* __restrict__ valuesA, const VALUE_TYPE* __restrict__ valuesB,
		const INDEX_TYPE nnzB, const INDEX_TYPE rowsB,
		LocalMap& __restrict__ lMap,
		INDEX_TYPE resultNnz, INDEX_TYPE minCol)
	{
		if (direct == true)
			iterateNumeric(startId, lastIdExcl, shift, true)
		else
			iterateNumeric(startId, lastIdExcl, shift, false)
	}
};

// this class creates all instances of my iterateMatrix template and selects the best instance for the specific call 'call'
template <typename INDEX_TYPE, uint32_t THREADS, class LocalMap>
class IterateMatrixCountingSingleRowMethods
{
#define iterateCounting(startId, lastIdExcl, shift, direct) call_template<direct>( \
	startId, lastIdExcl, shift,  \
	rowOffsetsB, colIdsA, colIdsB,\
	lMap, minCol);

#define iterate_templ_Counting(startId, lastIdExcl, shift, direct) iterateMatrixCountingSingleRow<INDEX_TYPE, THREADS, direct, shift>(	\
	startId, lastIdExcl,                                                                                                                            \
	rowOffsetsB, colIdsA, colIdsB ,		                                                                                                            \
	lMap, minCol);

private:
	template<bool DIRECT>
	__device__ static __forceinline__ void call_template(
		INDEX_TYPE startId, INDEX_TYPE lastIdExcl, uint32_t shift,
		const INDEX_TYPE* __restrict__ rowOffsetsB,
		const INDEX_TYPE* __restrict__ colIdsA, const INDEX_TYPE* __restrict__ colIdsB,
		const LocalMap& __restrict__ lMap,
		INDEX_TYPE minCol)
	{
		if (shift == 0)
			iterate_templ_Counting(startId, lastIdExcl, 0, DIRECT)
		else if (shift == 1)
			iterate_templ_Counting(startId, lastIdExcl, 1, DIRECT)
		else if (shift == 2)
			iterate_templ_Counting(startId, lastIdExcl, 2, DIRECT)
		else if (shift == 3)
			iterate_templ_Counting(startId, lastIdExcl, 3, DIRECT)
		else if (shift == 4)
			iterate_templ_Counting(startId, lastIdExcl, 4, DIRECT)
		else if (shift == 5)
			iterate_templ_Counting(startId, lastIdExcl, 5, DIRECT)
		else if (shift == 6)
			iterate_templ_Counting(startId, lastIdExcl, 6, DIRECT)
		else if (shift == 7)
			iterate_templ_Counting(startId, lastIdExcl, 7, DIRECT)
		else if (shift == 8)
			iterate_templ_Counting(startId, lastIdExcl, 8, DIRECT)
		else if (shift == 9)
			iterate_templ_Counting(startId, lastIdExcl, 9, DIRECT)
		else if (shift == 10)
			iterate_templ_Counting(startId, lastIdExcl, 10, DIRECT)
	}

public:
	__device__ static __forceinline__ void call(INDEX_TYPE startId, INDEX_TYPE lastIdExcl, int shift, bool direct,
		const INDEX_TYPE* __restrict__ rowOffsetsB,
		const INDEX_TYPE* __restrict__ colIdsA, const INDEX_TYPE* __restrict__ colIdsB,
		const LocalMap& __restrict__ lMap,
		INDEX_TYPE minCol)
	{
		if (direct == true)
			iterateCounting(startId, lastIdExcl, shift, true)
		else
			iterateCounting(startId, lastIdExcl, shift, false)
	}
};

// if the current row of A has only a single member, we can simply copy the corresponding row in B into C
template <typename INDEX_TYPE, typename VALUE_TYPE, uint32_t THREADS>
__device__ __forceinline__ void directSpGEMMNumericImplementation(
	const INDEX_TYPE* __restrict__ rowOffsetsB, const INDEX_TYPE* __restrict__ colIdsA, const INDEX_TYPE* __restrict__ colIdsB,
	const VALUE_TYPE* __restrict__ valuesA, const VALUE_TYPE* __restrict__ valuesB,
	INDEX_TYPE* __restrict__ colIdsC, VALUE_TYPE* __restrict__ valuesC,
	INDEX_TYPE resultStartId, INDEX_TYPE resultNnz,
	const INDEX_TYPE startId, const INDEX_TYPE lastIdExcl, bool setSortedBit)
{
	if (lastIdExcl <= startId) // no work for this block
		return;
	
	// if there is only one column in A, we can just directly copy the already sorted columns from B
	// no shared memory, no hash map, no sorting
	INDEX_TYPE rowB = colIdsA[startId];
	INDEX_TYPE idB = rowOffsetsB[rowB];
	VALUE_TYPE valA = valuesA[startId];

	for (INDEX_TYPE i = threadIdx.x; i < resultNnz; i += THREADS)
	{
		colIdsC[resultStartId + i] = colIdsB[idB + i];
		valuesC[resultStartId + i] = valuesB[idB + i] * valA;
	}

	// this tells the sorting kernel that the row is already sorted
	if (setSortedBit && threadIdx.x == 0)
		markRowSorted(colIdsC[resultStartId]);
}

// if the current row of A has only a single member, we can simply count the elements in the corresponding row in B
template <typename INDEX_TYPE, uint32_t THREADS>
__device__ __forceinline__ void directSpGEMMCountImplementation(
	const INDEX_TYPE* __restrict__ rowOffsetsB, const INDEX_TYPE* __restrict__ colIdsA, const INDEX_TYPE* __restrict__ colIdsB,
	INDEX_TYPE* __restrict__ rowOffsetsC, const INDEX_TYPE startRow, const INDEX_TYPE rowsB, const INDEX_TYPE nnzB,
	const INDEX_TYPE startId, const INDEX_TYPE lastIdExcl, INDEX_TYPE* __restrict__ maxNnzRow)
{
	if (lastIdExcl <= startId || threadIdx.x > 0) // no work for this block
		return;
	
	// if there is only one column in A, we can just directly count the elements of the single row in B

	INDEX_TYPE rowB = colIdsA[startId];
	INDEX_TYPE startIdB = rowOffsetsB[rowB];
	INDEX_TYPE lastIdBExcl = rowB + 1 < rowsB ? rowOffsetsB[rowB + 1]: nnzB;
	INDEX_TYPE nnz = lastIdBExcl - startIdB;
	rowOffsetsC[startRow] = nnz;
	atomicMax(maxNnzRow, nnz);
}

template <typename INDEX_TYPE, typename VALUE_TYPE, uint32_t SHARED_HASH_SIZE, uint32_t THREADS>
__device__ __forceinline__ void hashSpGEMMNumericImplementation(
	const INDEX_TYPE nnzA, const INDEX_TYPE nnzB, const INDEX_TYPE rowsA, const INDEX_TYPE rowsB,
	const INDEX_TYPE *__restrict__ rowOffsetsA, const INDEX_TYPE *__restrict__ rowOffsetsB, 
	const INDEX_TYPE *__restrict__ colIdsA, const INDEX_TYPE *__restrict__ colIdsB,
	const VALUE_TYPE *__restrict__ valuesA, const VALUE_TYPE *__restrict__ valuesB,
	INDEX_TYPE *__restrict__ colIdsC, VALUE_TYPE *__restrict__ valuesC,
	const INDEX_TYPE *__restrict__ rowOperations,
	Config::SortModes sortColumns, 
	uint32_t *rowMaxOperations, INDEX_TYPE startRow, INDEX_TYPE lastRowExcl, 
	INDEX_TYPE resultStartId, INDEX_TYPE resultNnz,
	INDEX_TYPE minCol, INDEX_TYPE maxCol, 
	const INDEX_TYPE startId, const INDEX_TYPE lastIdExcl, bool setSortedBit,
	uint32_t rowsPerBlock)
{
	if (lastIdExcl <= startId || resultNnz == 0) // no work for this block
		return;

	typedef HashMap<INDEX_TYPE, VALUE_TYPE> LocalMap;
	const int mapSize = SHARED_HASH_SIZE / (sizeof(INDEX_TYPE) + sizeof(VALUE_TYPE));
	const uint32_t elementsPerThread = (mapSize + THREADS - 1) / THREADS > 0 ? (mapSize + THREADS - 1) / THREADS : 1;

	typedef IterateMatrixNumericMethods<INDEX_TYPE, VALUE_TYPE, mapSize, THREADS, LocalMap> IterateMatrix;
	typedef cub::BlockRadixSort<INDEX_TYPE, THREADS, elementsPerThread, VALUE_TYPE> RadixSort;

	struct HashSmem
	{
		uint32_t countWrittenCols;
		uint32_t sumOperations;
		uint32_t maxOperationsPerCol;
		uint32_t shift;
	};

	LocalMap lMap;
	__shared__ HashSmem sMem;
	extern __shared__ int dynamicMem[];

	if (threadIdx.x == 0)
	{
		sMem.countWrittenCols = 0;
	}

	lMap.values = (VALUE_TYPE *)&dynamicMem[0];
	lMap.ids = (INDEX_TYPE *)&lMap.values[mapSize];
	lMap.capacity = mapSize;

	// figure out the best amount of threads per row in B
	// sum up the number of products and get the maximum using the first warp since the the number of rows never exceeds 32
	if (threadIdx.x < 32)
	{
		// get the amount of operations
		INDEX_TYPE rowOps = startRow + threadIdx.x < lastRowExcl ? rowOperations[startRow + threadIdx.x] : 0;
		if (lastRowExcl - startRow > 1)
		{
			for (int i = 16; i > 0; i /= 2)
				rowOps += __shfl_down_sync(0xFFFFFFFF, rowOps, i);
		}

		if (threadIdx.x == 0)
			sMem.sumOperations = rowOps;

		// get the maximum amount of operations per row in B accessed by this block
		if (rowMaxOperations != nullptr)
		{
			INDEX_TYPE rowMaxOps = startRow + threadIdx.x < lastRowExcl ? rowMaxOperations[startRow + threadIdx.x] : 0;

			if (lastRowExcl - startRow > 1)
			{
				for (int i = 16; i > 0; i /= 2)
					rowMaxOps = max(rowMaxOps, __shfl_down_sync(0xFFFFFFFF, rowMaxOps, i));
			}

			if (threadIdx.x == 0)
				sMem.maxOperationsPerCol = rowMaxOps;
		}

		// calculate the actual amount of threads per row in B
		if (threadIdx.x == 0)
		{
			sMem.shift = getThreadShiftNew(sMem.sumOperations, sMem.maxOperationsPerCol, 0, 31 - __clz(THREADS), lastIdExcl - startId);
		}
	}

	lMap.init();

	__syncthreads();

	const bool colFitInShared = (maxCol - minCol) < mapSize;
	bool direct = lastRowExcl - startRow == 1 && colFitInShared;

	IterateMatrix::call(startId, lastIdExcl, sMem.shift, direct, startRow, lastRowExcl,
						rowOffsetsA, rowOffsetsB, colIdsA, colIdsB, valuesA, valuesB, nnzA, nnzB, rowsA, rowsB,
						lMap, resultNnz, minCol);

	__syncthreads();

	INDEX_TYPE colOffset = direct ? minCol : 0;

	// write result to global
	// if inplace sorting is active or the current block has only small number of nz to sort
	// compaction -> all non-zero elements need to be at the front of the array for fast sorting

	if (sortColumns == Config::SortModes::InPlace) {
	#pragma unroll
		for (uint32_t i = 0; i < mapSize; i += THREADS)
		{
			uint32_t index = i + threadIdx.x;
			bool valid = index < mapSize;
			INDEX_TYPE id = valid ? lMap.ids[index] : 0;
			VALUE_TYPE val = valid ? lMap.values[index] : 0;

			if (valid && id != lMap.UNUSED())
				index = atomicAdd(&sMem.countWrittenCols, 1);

			__syncthreads();
			if (valid && id != lMap.UNUSED())
			{
				lMap.ids[index] = id;
				lMap.values[index] = val;
			}
		}

		__syncthreads();

		// Sort array by counting the entries with smaller colId
		for (int j = threadIdx.x; j < resultNnz; j += THREADS)
		{
			INDEX_TYPE target = lMap.ids[j];
			INDEX_TYPE count = 0;
			for (int k = 0; k < resultNnz; k++)
			{
				if (lMap.ids[k] < target)
					++count;
			}

			colIdsC[resultStartId + count] = lMap.idToCol(target) + colOffset;
			valuesC[resultStartId + count] = lMap.values[j];
		}
	} else {
		for (int j = threadIdx.x; j < resultNnz; j += THREADS) {
			if (lMap.ids[j] != lMap.UNUSED()) {
				INDEX_TYPE index = atomicAdd(&sMem.countWrittenCols, 1);
				colIdsC[resultStartId + index] = lMap.idToCol(lMap.ids[j]) + colOffset;
				valuesC[resultStartId + index] = lMap.values[j];
			}
		}
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE, uint32_t SHARED_HASH_SIZE, uint32_t THREADS>
__device__ __forceinline__ void hashSpGEMMNumericSingleRowImplementation(
	const INDEX_TYPE nnzB, const INDEX_TYPE rowsB,
	const INDEX_TYPE *__restrict__ rowOffsetsB, 
	const INDEX_TYPE *__restrict__ colIdsA, const INDEX_TYPE *__restrict__ colIdsB,
	const VALUE_TYPE *__restrict__ valuesA, const VALUE_TYPE *__restrict__ valuesB,
	INDEX_TYPE *__restrict__ colIdsC, VALUE_TYPE *__restrict__ valuesC,
	const INDEX_TYPE *__restrict__ rowOperations,
	Config::SortModes sortColumns,
	uint32_t *rowMaxOperations, INDEX_TYPE startRow, INDEX_TYPE lastRowExcl, 
	INDEX_TYPE resultStartId, INDEX_TYPE resultNnz,
	INDEX_TYPE minCol, INDEX_TYPE maxCol, const INDEX_TYPE startId, const INDEX_TYPE lastIdExcl, 
	bool setSortedBit, uint32_t rowsPerBlock)
{
	if (lastIdExcl <= startId || resultNnz == 0) // no work for this block
		return;

	typedef HashMap<INDEX_TYPE, VALUE_TYPE> LocalMap;
	const int mapSize = SHARED_HASH_SIZE / (sizeof(INDEX_TYPE) + sizeof(VALUE_TYPE));
	const uint32_t elementsPerThread = (mapSize + THREADS - 1) / THREADS > 0 ? (mapSize + THREADS - 1) / THREADS : 1;

	typedef IterateMatrixNumericSingleRowMethods<INDEX_TYPE, VALUE_TYPE, THREADS, LocalMap> IterateMatrix;
	typedef cub::BlockRadixSort<INDEX_TYPE, THREADS, elementsPerThread, VALUE_TYPE> RadixSort;

	struct HashSmem
	{
		uint32_t countWrittenCols;
		uint32_t shift;
	};

	__shared__ HashSmem sMem;
	extern __shared__ int dynamicMem[];

	LocalMap lMap;
	lMap.values = (VALUE_TYPE *)&dynamicMem[0];
	lMap.ids = (INDEX_TYPE *)&lMap.values[mapSize];
	lMap.capacity = mapSize;

	// figure out the best number of threads per row in B
	// sum up the number of products and get the maximum using the first warp since the the amount of rows never exceeds 32
	if (threadIdx.x == 0)
	{
		sMem.countWrittenCols = 0;
		// get the amount of operations
		INDEX_TYPE rowOps = rowOperations[startRow];

		// get the maximum amount of operations per row in B accessed by this block
		INDEX_TYPE rowMaxOps;
		if (rowMaxOperations != nullptr)
			rowMaxOps = rowMaxOperations[startRow];
		else
			rowMaxOps = rowOps / (lastIdExcl - startId);

		// calculate the actual amount of threads per row in B
		sMem.shift = getThreadShiftNew(rowOps, rowMaxOps, 0, 31 - __clz(THREADS), lastIdExcl - startId);
	}

	lMap.init();

	__syncthreads();

	bool direct = (maxCol - minCol) < mapSize;

	IterateMatrix::call(startId, lastIdExcl, sMem.shift, direct,
						rowOffsetsB, colIdsA, colIdsB, valuesA, valuesB, nnzB, rowsB,
						lMap, resultNnz, minCol);
	__syncthreads();

	INDEX_TYPE colOffset = direct ? minCol : 0;

	// write result to global

	// if inplace sorting is active or the current block has only small number of nz to sort
	if ((sortColumns == Config::InPlace || resultNnz < 500) && sortColumns != Config::SortModes::None)
	{
		// compaction -> all non-zero elements need to be at the front of the array for fast sorting
#pragma unroll
		for (uint32_t i = 0; i < mapSize; i += THREADS)
		{
			uint32_t index = i + threadIdx.x;
			bool valid = index < mapSize;
			INDEX_TYPE id = valid ? lMap.ids[index] : 0;
			VALUE_TYPE val = valid ? lMap.values[index] : 0;

			if (valid && id != lMap.UNUSED())
				index = atomicAdd(&sMem.countWrittenCols, 1);

			__syncthreads();
			if (valid && id != lMap.UNUSED())
			{
				lMap.ids[index] = id;
				lMap.values[index] = val;
			}
		}

		__syncthreads();

		// Sort array by counting the entries with smaller colId
		for (int j = threadIdx.x; j < resultNnz; j += THREADS)
		{
			INDEX_TYPE target = lMap.ids[j];
			INDEX_TYPE count = 0;
			for (int k = 0; k < resultNnz; k++)
			{
				if (lMap.ids[k] < target)
					++count;
			}

			colIdsC[resultStartId + count] = target + colOffset;
			valuesC[resultStartId + count] = lMap.values[j];
		}
	}
	else
	{
		// No sorting
		INDEX_TYPE index;
		for (int j = threadIdx.x; j < lMap.getSize(); j += THREADS)
		{
			if (lMap.ids[j] != lMap.UNUSED())
			{
				index = atomicAdd(&sMem.countWrittenCols, 1);
				colIdsC[resultStartId + index] = lMap.ids[j] + colOffset;
				valuesC[resultStartId + index] = lMap.values[j];
			}
		}
	}
}

// gets information about the work for this block and then calls the actual implementation of hashSpGEMM
template <typename INDEX_TYPE, typename VALUE_TYPE, class GlobalMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
__global__ void 
__launch_bounds__(1024, 2) 
hashSpGEMMNumeric(
	const INDEX_TYPE nnzA, const INDEX_TYPE nnzB, const INDEX_TYPE rowsA, const INDEX_TYPE rowsB, const INDEX_TYPE colsB,
	const INDEX_TYPE *__restrict__ rowOffsetsA, const INDEX_TYPE *__restrict__ rowOffsetsB, const INDEX_TYPE *__restrict__ colIdsA, const INDEX_TYPE *__restrict__ colIdsB,
	const VALUE_TYPE *__restrict__ valuesA, const VALUE_TYPE *__restrict__ valuesB,
	GlobalMap *__restrict__ maps, INDEX_TYPE mapCount, INDEX_TYPE *__restrict__ matCColIds,
	VALUE_TYPE *__restrict__ matCValues, const INDEX_TYPE *__restrict__ rowOffsetsC, INDEX_TYPE matCNnz, INDEX_TYPE *__restrict__ blockStartRow,
	const INDEX_TYPE *__restrict__ rowOperations,
	Config::SortModes sortColumns, const uint32_t numberBlocks, const INDEX_TYPE *__restrict__ rowColMinMax,
	uint32_t *rowMaxOperations, bool setSortedBit, uint32_t rowsPerBlock)
{
	INDEX_TYPE startRow = blockStartRow != nullptr ? blockStartRow[blockIdx.x] : blockIdx.x * rowsPerBlock;
	INDEX_TYPE lastRowExcl;
	if (blockStartRow != nullptr)
	{
		lastRowExcl = blockRangeToNumRows(startRow) + blockRangeToStartRow(startRow);
		startRow = blockRangeToStartRow(startRow);
	}
	else
		lastRowExcl = blockIdx.x + 1 < numberBlocks ? (blockStartRow != nullptr ? blockStartRow[blockIdx.x + 1] : (blockIdx.x + 1) * rowsPerBlock) : rowsA;
	lastRowExcl = min(lastRowExcl, rowsA);

	const INDEX_TYPE startId = rowOffsetsA[startRow];
	const INDEX_TYPE lastIdExcl = lastRowExcl < rowsA ? rowOffsetsA[lastRowExcl] : nnzA;
	const INDEX_TYPE resultStartId = rowOffsetsC[startRow];
	const INDEX_TYPE resultNnz = (lastRowExcl < rowsA ? rowOffsetsC[lastRowExcl] : matCNnz) - resultStartId;
	const INDEX_TYPE minCol = rowColMinMax != nullptr ? rowColMinMaxtoMinCol(rowColMinMax[startRow]) : 0;
	const INDEX_TYPE rowColSize = rowColMinMax != nullptr ? rowColMinMaxtoRowLength(rowColMinMax[startRow]) : spECK::numeric_limits<INDEX_TYPE>::max();
	const INDEX_TYPE colRange = min(colsB - minCol, rowColSize);

	if (lastRowExcl - startRow > 1 || SUPPORT_GLOBAL) {
		hashSpGEMMNumericImplementation<INDEX_TYPE, VALUE_TYPE, SHARED_HASH_SIZE, THREADS>(
			nnzA, nnzB, rowsA, rowsB, rowOffsetsA, rowOffsetsB, colIdsA, colIdsB, valuesA, valuesB,
			matCColIds, matCValues, rowOperations, sortColumns,
			rowMaxOperations, startRow, lastRowExcl, resultStartId, resultNnz, 
			minCol, minCol + colRange, startId, lastIdExcl, setSortedBit,
			rowsPerBlock);
	} else {
		hashSpGEMMNumericSingleRowImplementation<INDEX_TYPE, VALUE_TYPE, SHARED_HASH_SIZE, THREADS>(
			nnzB, rowsB, 
			rowOffsetsB, colIdsA, colIdsB, valuesA, valuesB,
			matCColIds, matCValues, rowOperations, sortColumns,
			rowMaxOperations, startRow, lastRowExcl, resultStartId, 
			resultNnz, minCol, minCol + colRange, startId, lastIdExcl, 
			setSortedBit, rowsPerBlock);
	}
}

template <typename INDEX_TYPE, unsigned MAX_ROWS_PER_BLOCK, class GlobalMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
__device__ __forceinline__ void hashSpGEMMCountImplementation(
	const INDEX_TYPE nnzA, const INDEX_TYPE nnzB, const INDEX_TYPE rowsA, const INDEX_TYPE rowsB,
	const INDEX_TYPE *__restrict__ rowOffsetsA, const INDEX_TYPE *__restrict__ rowOffsetsB, const INDEX_TYPE *__restrict__ colIdsA, const INDEX_TYPE *__restrict__ colIdsB,
	GlobalMap *__restrict__ maps, INDEX_TYPE mapCount, INDEX_TYPE *__restrict__ matCNnzRow, const INDEX_TYPE *__restrict__ rowOperations,
	uint32_t *__restrict__ rowMaxOperations, uint32_t *__restrict__ maxNnzPerRow,
	const INDEX_TYPE startId, const INDEX_TYPE lastIdExcl, const INDEX_TYPE startRow, const INDEX_TYPE lastRowExcl,
	const INDEX_TYPE minCol, const INDEX_TYPE maxCol, uint32_t rowsPerBlock)
{
	typedef HashMapNoValue<INDEX_TYPE, MAX_ROWS_PER_BLOCK> LocalMapNoValue;

	const int mapSize = SHARED_HASH_SIZE / sizeof(INDEX_TYPE) - MAX_ROWS_PER_BLOCK - 1;
	typedef IterateMatrixCountMethods<INDEX_TYPE, mapSize, SUPPORT_GLOBAL, THREADS, GlobalMap, LocalMapNoValue> IterateMatrix;

	extern __shared__ int dynamicShared[];
	struct HashSmem
	{
		GlobalMap *gMap;
		uint32_t sumOperations;
		uint32_t maxOperationsPerCol;
		uint32_t shift;
	};

	__shared__ HashSmem sMem;
	LocalMapNoValue lMapNoVal;
	lMapNoVal.occupancy = (INDEX_TYPE *)&dynamicShared[0];
	lMapNoVal.occupancyPerRow = &lMapNoVal.occupancy[1];
	lMapNoVal.ids = (INDEX_TYPE *)&lMapNoVal.occupancyPerRow[MAX_ROWS_PER_BLOCK];
	lMapNoVal.capacity = mapSize;

	if (startId >= lastIdExcl) {
		for (INDEX_TYPE rowA = startRow + threadIdx.x; rowA < lastRowExcl; rowA += THREADS)
			matCNnzRow[rowA] = 0;
		return;
	}

	if (threadIdx.x == 0)
	{
		sMem.gMap = nullptr;
	}

	if (threadIdx.x < 32) {
		INDEX_TYPE rowOps = startRow + threadIdx.x < lastRowExcl ? rowOperations[startRow + threadIdx.x] : 0;
		if (lastRowExcl - startRow > 1) {
			for (int i = 16; i > 0; i /= 2)
				rowOps += __shfl_down_sync(0xFFFFFFFF, rowOps, i);
		}

		if (threadIdx.x == 0)
			sMem.sumOperations = rowOps;

		if (rowMaxOperations != nullptr)
		{
			INDEX_TYPE rowMaxOps = startRow + threadIdx.x < lastRowExcl ? rowMaxOperations[startRow + threadIdx.x] : 0;

			if (lastRowExcl - startRow > 1) {
				for (int i = 16; i > 0; i /= 2)
					rowMaxOps = max(rowMaxOps, __shfl_down_sync(0xFFFFFFFF, rowMaxOps, i));
			}

			if (threadIdx.x == 0)
				sMem.maxOperationsPerCol = rowMaxOps;
		}

		if (threadIdx.x == 0) {
			sMem.shift = getThreadShiftNew(sMem.sumOperations, sMem.maxOperationsPerCol, 0, 31 - __clz(THREADS), lastIdExcl - startId);
		}
	}

	lMapNoVal.init(threadIdx.x == THREADS - 1);

	__syncthreads();

	bool supportGlobal = SUPPORT_GLOBAL;
	if (sMem.sumOperations < mapSize)
		supportGlobal = false;

	const bool colFitInShared = (maxCol - minCol) < mapSize;

	bool direct = !supportGlobal && lastRowExcl - startRow == 1 && colFitInShared;

	IterateMatrix::call(startId, lastIdExcl, sMem.shift, direct, startRow, lastRowExcl,
						rowOffsetsA, rowOffsetsB, colIdsA, colIdsB, nnzA, nnzB, rowsA, rowsB,
						supportGlobal, &sMem.gMap, lMapNoVal, maps, mapCount, sMem.sumOperations, minCol);

	__syncthreads();
	if (!supportGlobal || sMem.gMap == nullptr)
	{
		// all values fit into shared memory. only need to write shared memory occupancy

		for (int rowA = startRow + threadIdx.x; rowA < lastRowExcl; rowA += THREADS)
		{
			matCNnzRow[rowA] = lMapNoVal.occupancyPerRow[rowA - startRow];
		}

		if (threadIdx.x < 32)
		{
			uint32_t val = lMapNoVal.occupancyPerRow[threadIdx.x];
			for(int i = 16; i > 0; i/=2)
			{
				val = max(val, __shfl_down_sync(0xFFFFFFFF, val, i));
			}
			if (threadIdx.x == 0)
				atomicMax(maxNnzPerRow, val);
		}
	}
	else
	{
		// we used global map for this block -> write out current elements of shared memory first and save occupancy of global map
		saveSMemMapToGlobalNoVal(&lMapNoVal, sMem.gMap);
		__syncthreads();

		uint32_t occupancy = 0;
		for (int rowA = startRow + threadIdx.x; rowA < lastRowExcl; rowA += THREADS)
		{
			occupancy = sMem.gMap->occupancyPerRow[rowA - startRow];
			matCNnzRow[rowA] = occupancy;
		}

		// save the length of the longest row in A for load balancing
		if (threadIdx.x < 32)
		{
			for (int i = 16; i > 0; i /= 2)
			{
				occupancy = max(occupancy, __shfl_down_sync(0xFFFFFFFF, occupancy, i));
			}
			if (threadIdx.x == 0)
				atomicMax(maxNnzPerRow, occupancy);
		}

		__syncthreads();
		sMem.gMap->init(threadIdx.x == 0);
		__syncthreads();
		// DO NOT touch this threadfence. Or else matrix is set free, but the initialization might not yet be finished
		__threadfence();

		if (threadIdx.x == 0)
			freeMap(sMem.gMap);
	}
}

template <typename INDEX_TYPE, uint32_t SHARED_HASH_SIZE, uint32_t THREADS>
__device__ __forceinline__ void hashSpGEMMCountSingleRowImplementation(
	const INDEX_TYPE *__restrict__ rowOffsetsB, 
	const INDEX_TYPE *__restrict__ colIdsA, const INDEX_TYPE *__restrict__ colIdsB,
	INDEX_TYPE *__restrict__ matCNnzRow, const INDEX_TYPE *__restrict__ rowOperations,
	const uint32_t *__restrict__ rowMaxOperations, 
	uint32_t *__restrict__ maxNnzPerRow,
	const INDEX_TYPE startId, const INDEX_TYPE lastIdExcl, const INDEX_TYPE startRow,
	const INDEX_TYPE minCol, const INDEX_TYPE maxCol)
{
	typedef HashMapNoValue<INDEX_TYPE, 1> LocalMapNoValue;

	const int mapSize = SHARED_HASH_SIZE / sizeof(INDEX_TYPE) - 2;
	typedef IterateMatrixCountingSingleRowMethods<INDEX_TYPE, THREADS, LocalMapNoValue> IterateMatrix;

	if (startId >= lastIdExcl) {
		if (threadIdx.x == 0)
			matCNnzRow[startRow] = 0;
		return;
	}

	extern __shared__ int dynamicShared[];

	__shared__ uint32_t shift;
	LocalMapNoValue lMapNoVal;

	lMapNoVal.occupancy = (INDEX_TYPE *)&dynamicShared[0];
	lMapNoVal.occupancyPerRow = &lMapNoVal.occupancy[1];
	lMapNoVal.ids = (INDEX_TYPE *)&lMapNoVal.occupancyPerRow[1];
	lMapNoVal.capacity = mapSize;

	if (threadIdx.x == 0) {
		INDEX_TYPE rowOps = rowOperations[startRow];
	
		INDEX_TYPE rowMaxOps = rowOps / (lastIdExcl - startId);
		if (rowMaxOperations != nullptr)
		{
			rowMaxOps = rowMaxOperations[startRow];
		}
		shift = getThreadShiftNew(rowOps, rowMaxOps, 0, 31 - __clz(THREADS), lastIdExcl - startId);
	}

	lMapNoVal.init(threadIdx.x == THREADS - 1);

	__syncthreads();

	bool direct = (maxCol - minCol) < mapSize;

	IterateMatrix::call(startId, lastIdExcl, shift, direct,
						rowOffsetsB, colIdsA, colIdsB,
						lMapNoVal, minCol);

	__syncthreads();

	if (threadIdx.x == 0)
	{
		matCNnzRow[startRow] = lMapNoVal.occupancyPerRow[0];
		atomicMax(maxNnzPerRow, lMapNoVal.occupancyPerRow[0]);
	}
}

// gets information about the work for this block and then calls the actual implementation of hashSpGEMM
template <typename INDEX_TYPE, unsigned MAX_ROWS_PER_BLOCK, class GlobalMap, uint32_t SHARED_MEM_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
__global__ void 
__launch_bounds__(1024, 2) 
hashSpGEMMCount(
	const INDEX_TYPE nnzA, const INDEX_TYPE nnzB, const INDEX_TYPE rowsA, const INDEX_TYPE rowsB, const INDEX_TYPE colsB,
	const INDEX_TYPE *__restrict__ rowOffsetsA, const INDEX_TYPE *__restrict__ rowOffsetsB, const INDEX_TYPE *__restrict__ colIdsA, const INDEX_TYPE *__restrict__ colIdsB,
	GlobalMap *__restrict__ maps, INDEX_TYPE mapCount, INDEX_TYPE *__restrict__ matCNnzRow, const INDEX_TYPE *__restrict__ rowOperations,
	const INDEX_TYPE *__restrict__ blockStartRow, uint32_t numberBlocks, const INDEX_TYPE *__restrict__ rowColMinMax,
	uint32_t *__restrict__ rowMaxOperations, uint32_t *__restrict__ maxNnzPerRow, uint32_t rowsPerBlock)
{
	INDEX_TYPE startRow = blockStartRow != nullptr ? blockStartRow[blockIdx.x] : blockIdx.x * rowsPerBlock;
	INDEX_TYPE lastRowExcl;
	if (blockStartRow != nullptr)
	{
		lastRowExcl = blockRangeToNumRows(startRow) + blockRangeToStartRow(startRow);
		startRow = blockRangeToStartRow(startRow);
	}
	else
		lastRowExcl = blockIdx.x + 1 < numberBlocks ? (blockStartRow != nullptr ? blockStartRow[blockIdx.x + 1] : (blockIdx.x + 1) * rowsPerBlock) : rowsA;
	lastRowExcl = min(lastRowExcl, rowsA);

	const INDEX_TYPE startId = rowOffsetsA[startRow];
	const INDEX_TYPE lastIdExcl = lastRowExcl < rowsA ? rowOffsetsA[lastRowExcl] : nnzA;
	if (lastIdExcl <= startId)
	{
		if (threadIdx.x < lastRowExcl - startRow)
			matCNnzRow[startRow + threadIdx.x] = 0;

		return;
	}

	const INDEX_TYPE rowColSize = rowColMinMax != nullptr ? rowColMinMaxtoRowLength(rowColMinMax[startRow]) : spECK::numeric_limits<INDEX_TYPE>::max();
	const INDEX_TYPE minCol = rowColMinMax != nullptr ? rowColMinMaxtoMinCol(rowColMinMax[startRow]) : 0;
	const INDEX_TYPE colRange = min(colsB - minCol, rowColSize);

	if (lastRowExcl - startRow > 1 || (SUPPORT_GLOBAL && mapCount > 0)) {
		hashSpGEMMCountImplementation<INDEX_TYPE, MAX_ROWS_PER_BLOCK, GlobalMap, SHARED_MEM_SIZE, SUPPORT_GLOBAL, THREADS> (
			nnzA, nnzB, rowsA, rowsB, rowOffsetsA, rowOffsetsB, colIdsA, colIdsB,
			maps, mapCount, matCNnzRow, rowOperations, 
			rowMaxOperations, maxNnzPerRow, startId, lastIdExcl, startRow, lastRowExcl, 
			minCol, minCol + colRange, rowsPerBlock);
	}
	else 
	{
		hashSpGEMMCountSingleRowImplementation<INDEX_TYPE, SHARED_MEM_SIZE, THREADS>(
			rowOffsetsB, colIdsA, colIdsB,
			matCNnzRow, rowOperations,
			rowMaxOperations, maxNnzPerRow,
			startId, lastIdExcl, startRow,
			minCol, minCol + colRange);
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE, uint32_t MAX_ELEMENTS_BLOCK, bool SUPPORT_GLOBAL, uint32_t THREADS, uint32_t SHIFT, bool useRowOffsets>
__device__ __forceinline__ void iterateMatrixDenseNumeric(
	INDEX_TYPE startId, INDEX_TYPE lastIdExcl, uint32_t elementsPerBlock, INDEX_TYPE *rowCursor, 
	INDEX_TYPE rowsB, INDEX_TYPE nnzB, const INDEX_TYPE startRow,
	INDEX_TYPE resultStartId, const INDEX_TYPE* __restrict__ colIdsA, const INDEX_TYPE* __restrict__ colIdsB, 
	const INDEX_TYPE* __restrict__ rowOffsetsB,
	const VALUE_TYPE* __restrict__ valuesA, const VALUE_TYPE* __restrict__ valuesB,
	INDEX_TYPE* __restrict__ colIdsC, VALUE_TYPE* __restrict__ valuesC, const INDEX_TYPE minCol, const INDEX_TYPE maxCol,
	INDEX_TYPE* prefix, INDEX_TYPE* prefix2, 
	INDEX_TYPE* currentRowMinOffset, VALUE_TYPE* values, 
	void *temp_storage)
{
	typedef cub::BlockScan<INDEX_TYPE, THREADS> BlockScan;
	uint32_t colOffset = minCol;
	uint32_t accumulatedPrefix = 0;

	while (colOffset < maxCol)
	{
		for (int i = threadIdx.x; i < elementsPerBlock; i += THREADS)
		{
			values[i] = 0.0f;
		}

		for (int i = threadIdx.x; i < elementsPerBlock / 32 + 1; i += THREADS)
		{
			prefix[i] = 0;
		}
		__syncthreads();

		if (useRowOffsets) {
			for (INDEX_TYPE _idA = startId; _idA < lastIdExcl; _idA += THREADS >> SHIFT)
			{
				const INDEX_TYPE idA = _idA + (threadIdx.x >> SHIFT);
				const bool valid = idA < lastIdExcl;
				const INDEX_TYPE colA = valid ? colIdsA[idA] : 0;
				const INDEX_TYPE rowStart = valid ? rowOffsetsB[colA] : 0;
				const INDEX_TYPE startIdB = valid ? rowStart + (useRowOffsets ? rowCursor[idA - startId] : 0) : spECK::numeric_limits<INDEX_TYPE>::max();
				const INDEX_TYPE lastIdBExcl = valid ? (min(startIdB + elementsPerBlock, colA + 1 < rowsB ? rowOffsetsB[colA + 1] : nnzB)) : 0;
				const VALUE_TYPE valueA = valid ? valuesA[idA] : 0;
				
				if (threadIdx.x % (1 << SHIFT) == 0 && valid)
				{
					currentRowMinOffset[threadIdx.x >> SHIFT] = lastIdBExcl - rowStart;
				}

				__syncthreads();

				for (INDEX_TYPE idB = startIdB + (threadIdx.x % (1 << SHIFT)); idB < lastIdBExcl; idB += 1 << SHIFT)
				{
					auto colB = colIdsB[idB] - colOffset;
					if (colB < elementsPerBlock)
					{
						atomicAdd(&values[colB], valueA * valuesB[idB]);
						atomicOr(&prefix[colB / 32], 1 << (colB % 32));
					} else {
						atomicMin(&currentRowMinOffset[threadIdx.x >> SHIFT], idB - rowStart);
						break;
					}
				}
				__syncthreads();
				if (threadIdx.x % (1 << SHIFT) == 0 && valid)
					rowCursor[idA - startId] = currentRowMinOffset[threadIdx.x >> SHIFT];
			}
		} else {
			for (INDEX_TYPE idA = startId + (threadIdx.x >> SHIFT); idA < lastIdExcl; idA += THREADS >> SHIFT)
			{
				const INDEX_TYPE colA = colIdsA[idA];
				const INDEX_TYPE rowStart = rowOffsetsB[colA];
				const INDEX_TYPE startIdB = rowStart;
				const INDEX_TYPE lastIdBExcl = (min(startIdB + elementsPerBlock, colA + 1 < rowsB ? rowOffsetsB[colA + 1] : nnzB));
				const VALUE_TYPE valueA = valuesA[idA];

				for (INDEX_TYPE idB = startIdB + (threadIdx.x % (1 << SHIFT)); idB < lastIdBExcl; idB += 1 << SHIFT)
				{
					auto colB = colIdsB[idB] - colOffset;
					if (colB < elementsPerBlock)
					{
						atomicAdd(&values[colB], valueA * valuesB[idB]);
						atomicOr(&prefix[colB / 32], 1 << (colB % 32));
					}
				}
			}
		}

		__syncthreads();
		const uint32_t localElements = (MAX_ELEMENTS_BLOCK + THREADS - 1) / THREADS; // ceil -> (a + b - 1) / b;
		INDEX_TYPE thread_data[localElements];
		for (int i = 0; i < localElements; i++)
		{
			uint32_t id = threadIdx.x * localElements + i;
			thread_data[i] = id < elementsPerBlock / 32 + 1 ? __popc(prefix[id]) : 0U;
		}
		
		BlockScan(*((typename BlockScan::TempStorage*) temp_storage)).InclusiveSum(thread_data, thread_data);
		
		for (int i = 0; i < localElements; i++)
		{
			uint32_t id = threadIdx.x * localElements + i;
			if (id < elementsPerBlock / 32 + 1)
				prefix2[id] = thread_data[i];
		}
		__syncthreads();
		
		for (int i = threadIdx.x; i < elementsPerBlock; i += THREADS)
		{
			INDEX_TYPE warpPrefix = i / 32 == 0 ? 0 : prefix2[i / 32 - 1];
			bool isNonZero = prefix[i / 32] & (1 << (i % 32));
			warpPrefix += __popc(prefix[i / 32] & ((1 << (i % 32)) - 1));
			if (isNonZero)
			{
				uint32_t id = warpPrefix + accumulatedPrefix + resultStartId;
				colIdsC[id] = colOffset + i;
				valuesC[id] = values[i];
			}
		}

		colOffset += elementsPerBlock;
		if (colOffset >= maxCol) 
			return;

		accumulatedPrefix += prefix2[elementsPerBlock / 32];
		__syncthreads();
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE, class GlobalRowOffsetsMap, uint32_t SHARED_MEM_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
__device__ __forceinline__ void denseSpGEMMNumericImplementation(
	const INDEX_TYPE nnzB, const INDEX_TYPE rowsB,
	const INDEX_TYPE *__restrict__ rowOffsetsB, const INDEX_TYPE *__restrict__ colIdsA, const INDEX_TYPE *__restrict__ colIdsB,
	const VALUE_TYPE *__restrict__ valuesA, const VALUE_TYPE *__restrict__ valuesB,
	GlobalRowOffsetsMap *__restrict__ maps, INDEX_TYPE mapCount,
	INDEX_TYPE *__restrict__ colIdsC, VALUE_TYPE *__restrict__ valuesC,
	const INDEX_TYPE *__restrict__ rowOperations,
	uint32_t *rowMaxOperations, INDEX_TYPE startRow,
	INDEX_TYPE resultStartId, const INDEX_TYPE minCol, const INDEX_TYPE maxCol, const INDEX_TYPE startId, const INDEX_TYPE lastIdExcl, const bool setSortedBit,
	uint32_t rowsPerBlock)
{
	extern __shared__ int dynamicShared[];
	typedef cub::BlockScan<INDEX_TYPE, THREADS> BlockScan;

	struct SMEM
	{
		// this map contains the current cursor in all rows of B which are referenced by this block. only used if we need multiple iterations
		GlobalRowOffsetsMap *globalOffsetsMap;
		INDEX_TYPE *rowOffsets;
	};
	__shared__ SMEM sMem;
	VALUE_TYPE *values;
	INDEX_TYPE *prefix;
	INDEX_TYPE *prefix2;
	INDEX_TYPE *currentRowMinOffset;
	INDEX_TYPE *rowOffsets;
	typename BlockScan::TempStorage *temp_storage;

	// 32/64 bit for float/double + 1 prefix + 1 prefix2
	const uint32_t bitsPerElement = sizeof(VALUE_TYPE) * 8 + 2;

	// 32 indexType for currentRowMinOffset
	const INDEX_TYPE freeBytesBlock = (SHARED_MEM_SIZE - sizeof(BlockScan::TempStorage) - 32 * sizeof(INDEX_TYPE)) - 64; // just leave some space free
	const INDEX_TYPE maxElementsBlock = freeBytesBlock * 8 / bitsPerElement;
	INDEX_TYPE elementsPerBlock = maxElementsBlock;

	bool useGlobalOffsetsMap = (maxCol - minCol) < elementsPerBlock ? false : (lastIdExcl - startId) * sizeof(INDEX_TYPE) > freeBytesBlock * 3 / 4;
	bool useRowOffsets = true;

	if (useGlobalOffsetsMap)
	{
		if(threadIdx.x == 0)
		{
			sMem.globalOffsetsMap = reserveMap<GlobalRowOffsetsMap>(maps, mapCount);
			sMem.rowOffsets = sMem.globalOffsetsMap->ids;
		}
		__syncthreads();
		temp_storage = (typename  BlockScan::TempStorage*) (void*)dynamicShared;
		values = (VALUE_TYPE*) &((char*)dynamicShared)[sizeof(BlockScan::TempStorage)];
		prefix = (INDEX_TYPE*) &values[elementsPerBlock];
		prefix2 = (INDEX_TYPE*) &prefix[elementsPerBlock / 32 + 1];
		currentRowMinOffset = (INDEX_TYPE*) &prefix2[elementsPerBlock / 32 + 1];
		__syncthreads();
		__threadfence();
		rowOffsets = sMem.rowOffsets;

		for (int i = threadIdx.x; i < sMem.globalOffsetsMap->getSize(); ++i)
			rowOffsets[i] = 0;
	}
	else
	{
		uint32_t offsetElements = (maxCol - minCol) < elementsPerBlock ? 0 : lastIdExcl - startId + 1;
		useRowOffsets = offsetElements > 0;
		elementsPerBlock -= offsetElements * sizeof(INDEX_TYPE) * 8 / bitsPerElement;
		if (threadIdx.x == 0)
		{
			sMem.globalOffsetsMap = nullptr;
		}

		temp_storage = (typename  BlockScan::TempStorage*) (void*)dynamicShared;
		values = (VALUE_TYPE*) &((char*)dynamicShared)[sizeof(BlockScan::TempStorage)];
		prefix = (INDEX_TYPE*) &values[elementsPerBlock];
		prefix2 = (INDEX_TYPE*) &prefix[elementsPerBlock / 32 + 1];
		rowOffsets = (INDEX_TYPE*) &prefix2[elementsPerBlock / 32 + 1];
		currentRowMinOffset = (INDEX_TYPE*) &rowOffsets[offsetElements];

		for(int i = threadIdx.x; i < offsetElements; i+=THREADS)
		{
			rowOffsets[i] = 0;
		}
	}

	uint32_t shift = getThreadShiftNew(rowOperations[startRow], rowMaxOperations[startRow], 5U, 31U - __clz(THREADS), lastIdExcl - startId);

#define iterate(SHIFT, useRowOffsets) iterateMatrixDenseNumeric<INDEX_TYPE, VALUE_TYPE, maxElementsBlock, SUPPORT_GLOBAL, THREADS, SHIFT, useRowOffsets>\
	(startId, lastIdExcl, elementsPerBlock, rowOffsets,\
		rowsB, nnzB, startRow,\
		resultStartId, colIdsA, colIdsB,\
		rowOffsetsB,\
		valuesA, valuesB, \
		colIdsC, valuesC, minCol, maxCol,\
		prefix, prefix2, currentRowMinOffset, values, temp_storage)

	switch (shift)
	{
	case 10:
		useRowOffsets ? iterate(10, true) : iterate(10, false);
		break;
	case 9:
		useRowOffsets ? iterate(9, true) : iterate(9, false);
		break;
	case 8:
		useRowOffsets ? iterate(8, true) : iterate(8, false);
		break;
	case 7:
		useRowOffsets ? iterate(7, true) : iterate(7, false);
		break;
	case 6:
		useRowOffsets ? iterate(6, true) : iterate(6, false);
		break;
	default:
		useRowOffsets ? iterate(5, true) : iterate(5, false);
		break;
	}

	if (setSortedBit)
	{
		__syncthreads();

		// this tells the sorting kernel that the row is already sorted
		if (threadIdx.x == 0)
			markRowSorted(colIdsC[resultStartId]);
	}
	

	if(sMem.globalOffsetsMap != nullptr)
	{
		for (int i = threadIdx.x; i < sMem.globalOffsetsMap->getSize(); ++i)
			sMem.globalOffsetsMap->ids[i] = sMem.globalOffsetsMap->UNUSED();

		__syncthreads();
		__threadfence();
		if(threadIdx.x == 0)
			freeMap(sMem.globalOffsetsMap);
	}
}

// gets information about the work for this block and then calls the actual implementation of denseSpgemm
template <typename INDEX_TYPE, typename VALUE_TYPE, class GlobalOffsetMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
__global__ void __launch_bounds__(1024, 1) denseSpGEMMNumeric(
	const INDEX_TYPE nnzA, const INDEX_TYPE nnzB, const INDEX_TYPE rowsA, const INDEX_TYPE rowsB, const INDEX_TYPE colsB,
	const INDEX_TYPE *__restrict__ rowOffsetsA, const INDEX_TYPE *__restrict__ rowOffsetsB, const INDEX_TYPE *__restrict__ colIdsA, const INDEX_TYPE *__restrict__ colIdsB,
	const VALUE_TYPE *__restrict__ valuesA, const VALUE_TYPE *__restrict__ valuesB,
	GlobalOffsetMap *__restrict__ maps, INDEX_TYPE mapCount, INDEX_TYPE *__restrict__ colIdsC,
	VALUE_TYPE *__restrict__ valuesC, const INDEX_TYPE *__restrict__ rowOffsetsC, INDEX_TYPE matCNnz, INDEX_TYPE *__restrict__ blockStartRow,
	const INDEX_TYPE *__restrict__ rowOperations,
	const uint32_t numberBlocks, const INDEX_TYPE *__restrict__ rowColMinMax,
	uint32_t *rowMaxOperations, bool setSortedBit, uint32_t rowsPerBlock)
{
	INDEX_TYPE startRow = blockStartRow != nullptr ? blockStartRow[blockIdx.x] : blockIdx.x * rowsPerBlock;
	INDEX_TYPE lastRowExcl;
	if (blockStartRow != nullptr)
	{
		lastRowExcl = blockRangeToNumRows(startRow) + blockRangeToStartRow(startRow);
		startRow = blockRangeToStartRow(startRow);
	}
	else
		lastRowExcl = blockIdx.x + 1 < numberBlocks ? (blockStartRow != nullptr ? blockStartRow[blockIdx.x + 1] : (blockIdx.x + 1) * rowsPerBlock) : rowsA;
	lastRowExcl = min(lastRowExcl, rowsA);

	const INDEX_TYPE startId = rowOffsetsA[startRow];
	const INDEX_TYPE lastIdExcl = lastRowExcl < rowsA ? rowOffsetsA[lastRowExcl] : nnzA;
	const INDEX_TYPE resultStartId = rowOffsetsC[startRow];
	const INDEX_TYPE minCol = rowColMinMax != nullptr ? rowColMinMaxtoMinCol(rowColMinMax[startRow]) : 0;
	const INDEX_TYPE rowColSize = rowColMinMax != nullptr ? rowColMinMaxtoRowLength(rowColMinMax[startRow]) : spECK::numeric_limits<INDEX_TYPE>::max();
	const INDEX_TYPE colRange = min(colsB - minCol, rowColSize);

	denseSpGEMMNumericImplementation<INDEX_TYPE, VALUE_TYPE, GlobalOffsetMap, SHARED_HASH_SIZE, SUPPORT_GLOBAL, THREADS>(
		nnzB, rowsB, rowOffsetsB, colIdsA, colIdsB, valuesA, valuesB,
		maps, mapCount, colIdsC, valuesC, rowOperations,
		rowMaxOperations, startRow, resultStartId, minCol, minCol + colRange, startId, lastIdExcl, setSortedBit, rowsPerBlock);
}


template <typename INDEX_TYPE, uint32_t MAX_ELEMENTS_PER_BLOCK, bool SUPPORT_GLOBAL, uint32_t THREADS, uint32_t SHIFT, bool useRowOffsets>
__device__ __forceinline__ void iterateMatrixDenseCount(INDEX_TYPE startId, INDEX_TYPE lastIdExcl, uint32_t elementsPerBlock, INDEX_TYPE *rowCursor,
	INDEX_TYPE colsB, INDEX_TYPE rowsB, INDEX_TYPE nnzB,
	INDEX_TYPE startRow, const INDEX_TYPE* __restrict__ colIdsA, const INDEX_TYPE* __restrict__ colIdsB,
	const INDEX_TYPE* __restrict__ rowOffsetsB,
	INDEX_TYPE* __restrict__ rowOffsetsC, uint32_t *maxNnzPerRow, uint32_t minCol, uint32_t maxCol,
	INDEX_TYPE *prefix, INDEX_TYPE *currentRowMinOffset, void *temp_storage)
{
	typedef cub::BlockReduce<INDEX_TYPE, THREADS> BlockReduce;
	uint32_t colOffset = minCol;
	uint32_t accumulatedPrefix = 0;

	while (colOffset < maxCol)
	{
		for (int i = threadIdx.x; i < elementsPerBlock / 32 + 1; i += THREADS)
		{
			prefix[i] = 0;
		}
		__syncthreads();

		if (useRowOffsets) {
			for (INDEX_TYPE _idA = startId; _idA < lastIdExcl; _idA += THREADS >> SHIFT)
			{
				const INDEX_TYPE idA = _idA + (threadIdx.x >> SHIFT);
				const bool valid = idA < lastIdExcl;
				INDEX_TYPE colA = valid ? colIdsA[idA] : 0;
				INDEX_TYPE rowStart = valid ? rowOffsetsB[colA] : 0;
				INDEX_TYPE startIdB = valid ? rowStart + (useRowOffsets ? rowCursor[idA - startId] : 0) : spECK::numeric_limits<INDEX_TYPE>::max();
				INDEX_TYPE lastIdBExcl = valid ? (min(startIdB + elementsPerBlock, rowOffsetsB[colA + 1])) : 0;

				if (threadIdx.x % (1 << SHIFT) == 0 && valid)
				{
					currentRowMinOffset[threadIdx.x >> SHIFT] = lastIdBExcl - rowStart;
				}

				__syncthreads();

				for (INDEX_TYPE idB = startIdB + (threadIdx.x % (1 << SHIFT)); idB < lastIdBExcl; idB += 1 << SHIFT)
				{
					auto colB = colIdsB[idB] - colOffset;
					if (colB < elementsPerBlock)
					{
						atomicOr(&prefix[colB / 32], 1 << (colB % 32));
					}
					else if(useRowOffsets)
					{
						atomicMin(&currentRowMinOffset[threadIdx.x >> SHIFT], idB - rowStart);
						break;
					}
				}
				__syncthreads();
				if (threadIdx.x % (1 << SHIFT) == 0 && valid)
					rowCursor[idA - startId] = currentRowMinOffset[threadIdx.x >> SHIFT];
			}
		} else {
			for (INDEX_TYPE idA = startId + (threadIdx.x >> SHIFT); idA < lastIdExcl; idA += THREADS >> SHIFT)
			{
				INDEX_TYPE colA = colIdsA[idA];
				INDEX_TYPE rowStart = rowOffsetsB[colA];
				INDEX_TYPE startIdB = rowStart;
				INDEX_TYPE lastIdBExcl = min(startIdB + elementsPerBlock, rowOffsetsB[colA + 1]);

				for (INDEX_TYPE idB = startIdB + (threadIdx.x % (1 << SHIFT)); idB < lastIdBExcl; idB += 1 << SHIFT)
				{
					auto colB = colIdsB[idB] - colOffset;
					if (colB < elementsPerBlock)
					{
						atomicOr(&prefix[colB / 32], 1 << (colB % 32));
					}
				}
			}
		}

		__syncthreads();
		const uint32_t localElements = (MAX_ELEMENTS_PER_BLOCK / 32 + THREADS - 1) / THREADS; // ceil -> (a + b - 1) / b;
		INDEX_TYPE thread_data[localElements];
		for (int i = 0; i < localElements; i++)
		{
			uint32_t id = threadIdx.x * localElements + i;
			thread_data[i] = id <= elementsPerBlock / 32 ? __popc(prefix[id]) : 0U;
		}

		colOffset += elementsPerBlock;
		accumulatedPrefix += BlockReduce(*((typename BlockReduce::TempStorage*)temp_storage)).Sum(thread_data);

		if (colOffset < maxCol)
			__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		rowOffsetsC[startRow] = accumulatedPrefix;
		atomicMax(maxNnzPerRow, accumulatedPrefix);
	}
}

template <typename INDEX_TYPE, uint32_t SHARED_MEM_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
__device__ __forceinline__ void denseSpGEMMCountImplementation(
	const INDEX_TYPE nnzB, const INDEX_TYPE rowsB, const INDEX_TYPE colsB,
	const INDEX_TYPE *__restrict__ rowOffsetsB, const INDEX_TYPE *__restrict__ colIdsA, const INDEX_TYPE *__restrict__ colIdsB,
	void *__restrict__ maps, INDEX_TYPE mapCount,
	INDEX_TYPE *__restrict__ rowOffsetsC,
	const INDEX_TYPE *__restrict__ rowOperations,
	uint32_t *rowMaxOperations, uint32_t *maxNnzPerRow,
	const INDEX_TYPE startId, const INDEX_TYPE lastIdExcl, const INDEX_TYPE startRow,
	const INDEX_TYPE minCol, const INDEX_TYPE maxCol, uint32_t rowsPerBlock)
{
	if (lastIdExcl <= startId) // no work for this block
		return;

	extern __shared__ int dynamicShared[];
	typedef cub::BlockReduce<INDEX_TYPE, THREADS> BlockReduce;
	typedef HashMapNoValue<INDEX_TYPE, 1> GlobalOffsetMap;

	INDEX_TYPE *currentRowMinOffset;
	__shared__ GlobalOffsetMap *globalOffsetsMap;

	INDEX_TYPE *prefix;
	typename BlockReduce::TempStorage *temp_storage;
	INDEX_TYPE *rowOffsets;

	const INDEX_TYPE freeBytesBlock = (SHARED_MEM_SIZE - sizeof(BlockReduce::TempStorage)) - 32 * sizeof(INDEX_TYPE); // leave some space for rowOffsets
	const INDEX_TYPE maxElementsBlock = freeBytesBlock * 8; // max 1 col per bit
	INDEX_TYPE elementsPerBlock = maxElementsBlock - 32;

	bool useGlobalOffsetsMap = (maxCol - minCol) < elementsPerBlock ? false : (lastIdExcl - startId) * sizeof(INDEX_TYPE) > freeBytesBlock * 3 / 4;
	bool useRowOffsets = true;

	if (useGlobalOffsetsMap)
	{
		if(threadIdx.x == 0)
		{
			globalOffsetsMap = reserveMap<GlobalOffsetMap>((GlobalOffsetMap *)maps, mapCount);
		}
		__syncthreads();
		temp_storage = (typename  BlockReduce::TempStorage*) (void*)dynamicShared;
		currentRowMinOffset = (INDEX_TYPE*) &temp_storage[1];
		prefix = &currentRowMinOffset[32];
		rowOffsets = globalOffsetsMap->ids;
		__syncthreads();
		__threadfence();

		for (int i = threadIdx.x; i < globalOffsetsMap->getSize(); i += THREADS)
			rowOffsets[i] = 0;
	}
	else
	{
		uint32_t offsetElements = (maxCol - minCol) < elementsPerBlock ? 0 : lastIdExcl - startId + 1;
		useRowOffsets = offsetElements > 0;
		elementsPerBlock -= offsetElements * sizeof(INDEX_TYPE) * 8;

		temp_storage = (typename  BlockReduce::TempStorage*) (void*)dynamicShared;
		currentRowMinOffset = (INDEX_TYPE*) &temp_storage[1];
		prefix = &currentRowMinOffset[32];
		rowOffsets = (INDEX_TYPE*) &prefix[elementsPerBlock / 32 + 1];
		globalOffsetsMap = nullptr;

		for (int i = threadIdx.x; i < offsetElements; i += THREADS)
			rowOffsets[i] = 0;
	}

	uint32_t shift = getThreadShiftNew(rowOperations[startRow], rowMaxOperations[startRow], 5, 31 - __clz(THREADS), lastIdExcl - startId);

#define iterate(shift, useRowOffsets) iterateMatrixDenseCount<INDEX_TYPE, maxElementsBlock, SUPPORT_GLOBAL, THREADS, shift, useRowOffsets>\
	(startId, lastIdExcl, elementsPerBlock, rowOffsets,\
		colsB, rowsB, nnzB,\
		startRow, colIdsA, colIdsB,\
		rowOffsetsB,\
		rowOffsetsC, maxNnzPerRow, minCol, maxCol, \
		prefix, currentRowMinOffset, temp_storage)

	switch (shift)
	{
	case 10:
		useRowOffsets ? iterate(10, true) : iterate(10, false);
		break;
	case 9:
		useRowOffsets ? iterate(9, true) : iterate(9, false);
		break;
	case 8:
		useRowOffsets ? iterate(8, true) : iterate(8, false);
		break;
	case 7:
		useRowOffsets ? iterate(7, true) : iterate(7, false);
		break;
	case 6:
		useRowOffsets ? iterate(6, true) : iterate(6, false);
		break;
	default:
		useRowOffsets ? iterate(5, true) : iterate(5, false);
		break;		
	}

	if (globalOffsetsMap != nullptr)
	{
		for (int i = threadIdx.x; i < globalOffsetsMap->getSize(); ++i)
			globalOffsetsMap->ids[i] = globalOffsetsMap->UNUSED();

		__syncthreads();
		__threadfence();
		if (threadIdx.x == 0)
			freeMap(globalOffsetsMap);
	}
}

// gets information about the work for this block and then calls the actual implementation of denseSpgemm
template <typename INDEX_TYPE, class GlobalOffsetMap, uint32_t SHARED_MEM_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
__global__ void __launch_bounds__(1024, 1) denseSpGEMMCount(
	const INDEX_TYPE nnzA, const INDEX_TYPE nnzB, const INDEX_TYPE rowsA, const INDEX_TYPE rowsB, const INDEX_TYPE colsB,
	const INDEX_TYPE *__restrict__ rowOffsetsA, const INDEX_TYPE *__restrict__ rowOffsetsB, const INDEX_TYPE *__restrict__ colIdsA, const INDEX_TYPE *__restrict__ colIdsB,
	GlobalOffsetMap *__restrict__ maps, INDEX_TYPE mapCount,
	INDEX_TYPE *__restrict__ rowOffsetsC, INDEX_TYPE *__restrict__ blockStartRow,
	const INDEX_TYPE *__restrict__ rowOperations,
	const uint32_t numberBlocks, const INDEX_TYPE *__restrict__ rowColMinMax,
	uint32_t *rowMaxOperations, uint32_t *maxNnzPerRow, uint32_t rowsPerBlock)
{
	INDEX_TYPE startRow = blockStartRow != nullptr ? blockStartRow[blockIdx.x] : blockIdx.x * rowsPerBlock;
	INDEX_TYPE lastRowExcl;
	if (blockStartRow != nullptr)
	{
		lastRowExcl = blockRangeToNumRows(startRow) + blockRangeToStartRow(startRow);
		startRow = blockRangeToStartRow(startRow);
	}
	else
		lastRowExcl = blockIdx.x + 1 < numberBlocks ? (blockStartRow != nullptr ? blockStartRow[blockIdx.x + 1] : (blockIdx.x + 1) * rowsPerBlock) : rowsA;
	lastRowExcl = min(lastRowExcl, rowsA);

	const INDEX_TYPE startId = rowOffsetsA[startRow];
	const INDEX_TYPE lastIdExcl = lastRowExcl < rowsA ? rowOffsetsA[lastRowExcl] : nnzA;
	const INDEX_TYPE minCol = rowColMinMax != nullptr ? rowColMinMaxtoMinCol(rowColMinMax[startRow]) : 0;
	const INDEX_TYPE rowColSize = rowColMinMax != nullptr ? rowColMinMaxtoRowLength(rowColMinMax[startRow]) : spECK::numeric_limits<INDEX_TYPE>::max();
	const INDEX_TYPE colRange = min(colsB - minCol, rowColSize);

	denseSpGEMMCountImplementation<INDEX_TYPE, SHARED_MEM_SIZE, SUPPORT_GLOBAL, THREADS>(
		nnzB, rowsB, colsB, rowOffsetsB, colIdsA, colIdsB, maps, mapCount, rowOffsetsC,
		rowOperations, rowMaxOperations, maxNnzPerRow, startId, lastIdExcl, startRow, minCol, minCol + colRange, rowsPerBlock);
}

// gets information about the work for this block and then decides which method to use -> hash, dense or direct
template <typename INDEX_TYPE, typename VALUE_TYPE, class GlobalHashMap, class GlobalRowOffsetMap, uint32_t SHARED_MEM_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
__global__ void 
__launch_bounds__(1024, 2) 
spGEMMNumericLauncher(
	const INDEX_TYPE nnzA, const INDEX_TYPE nnzB, const INDEX_TYPE nnzC, const INDEX_TYPE rowsA, const INDEX_TYPE rowsB, const INDEX_TYPE colsB,
	const INDEX_TYPE *__restrict__ rowOffsetsA, const INDEX_TYPE *__restrict__ rowOffsetsB, const INDEX_TYPE *__restrict__ colIdsA, const INDEX_TYPE *__restrict__ colIdsB,
	const VALUE_TYPE *__restrict__ valuesA, const VALUE_TYPE *__restrict__ valuesB,
	GlobalHashMap *hashMaps, INDEX_TYPE hashMapCount, GlobalRowOffsetMap *rowOffsetMaps, INDEX_TYPE rowOffsetMapCount,
	INDEX_TYPE *__restrict__ colIdsC, VALUE_TYPE *__restrict__ valuesC,
	const INDEX_TYPE *__restrict__ rowOffsetsC, INDEX_TYPE *__restrict__ blockStartRow,
	const INDEX_TYPE *__restrict__ rowOperations,
	Config::SortModes sortColumns, const uint32_t numberBlocks, const INDEX_TYPE *__restrict__ rowColMinMax,
	uint32_t *rowMaxOperations, uint32_t minimumDensity, bool setSortedBit, uint32_t rowsPerBlock)
{
	INDEX_TYPE startRow = blockStartRow != nullptr ? blockStartRow[blockIdx.x] : blockIdx.x * rowsPerBlock;
	INDEX_TYPE lastRowExcl;
	if (blockStartRow != nullptr)
	{
		lastRowExcl = blockRangeToNumRows(startRow) + blockRangeToStartRow(startRow);
		startRow = blockRangeToStartRow(startRow);
	}
	else
		lastRowExcl = blockIdx.x + 1 < numberBlocks ? (blockStartRow != nullptr ? blockStartRow[blockIdx.x + 1] : (blockIdx.x + 1) * rowsPerBlock) : rowsA;
	lastRowExcl = min(lastRowExcl, rowsA);

	const INDEX_TYPE startId = rowOffsetsA[startRow];
	const INDEX_TYPE lastIdExcl = lastRowExcl < rowsA ? rowOffsetsA[lastRowExcl] : nnzA;
	const INDEX_TYPE resultStartId = rowOffsetsC[startRow];
	const INDEX_TYPE resultNnz = (lastRowExcl < rowsA ? rowOffsetsC[lastRowExcl] : nnzC) - resultStartId;
	const INDEX_TYPE minCol = rowColMinMax != nullptr ? rowColMinMaxtoMinCol(rowColMinMax[startRow]) : 0;
	const INDEX_TYPE rowColSize = rowColMinMax != nullptr ? rowColMinMaxtoRowLength(rowColMinMax[startRow]) : spECK::numeric_limits<INDEX_TYPE>::max();
	const INDEX_TYPE colRange = min(colsB - minCol, rowColSize);

	const int hashMapSize = SHARED_MEM_SIZE / (sizeof(INDEX_TYPE) + sizeof(VALUE_TYPE)) - THREADS;


	// TODO: check if variable can be removed. only the minimum density should be required
	int denseIterations = 1;
	{
		typedef cub::BlockScan<INDEX_TYPE, THREADS> BlockScan;
		const uint32_t bitsPerElement = sizeof(VALUE_TYPE) * 8 + 2;
		const INDEX_TYPE freeBytesBlock = (SHARED_MEM_SIZE - sizeof(BlockScan::TempStorage) - 32 * sizeof(INDEX_TYPE)) - 64; // leave some space free
		
		int maxElementsPerDenseIteration = freeBytesBlock * 8 / bitsPerElement;
		denseIterations += colRange / 2 / maxElementsPerDenseIteration;
	}

	if (lastIdExcl - startId == 1 && lastRowExcl - startRow == 1)
	{
		directSpGEMMNumericImplementation<INDEX_TYPE, VALUE_TYPE, THREADS>(
			rowOffsetsB, colIdsA, colIdsB, valuesA, valuesB, colIdsC, valuesC, resultStartId, resultNnz, startId, lastIdExcl, setSortedBit);
	}
	else if (lastRowExcl - startRow == 1 && (resultNnz >= hashMapSize || resultNnz * 100 > colRange * minimumDensity * denseIterations) && (lastIdExcl - startId) < (SHARED_MEM_SIZE / 2 / sizeof(INDEX_TYPE)))
	{
		denseSpGEMMNumericImplementation<INDEX_TYPE, VALUE_TYPE, GlobalRowOffsetMap, SHARED_MEM_SIZE, SUPPORT_GLOBAL, THREADS>(
			nnzB, rowsB, rowOffsetsB, colIdsA, colIdsB, valuesA, valuesB,
			rowOffsetMaps, rowOffsetMapCount, colIdsC, valuesC, rowOperations,
			rowMaxOperations, startRow, resultStartId, minCol, minCol + colRange,
			startId, lastIdExcl, setSortedBit, rowsPerBlock);
	}
	else 
	if (lastRowExcl - startRow == 1 && !SUPPORT_GLOBAL) 
	{
		hashSpGEMMNumericSingleRowImplementation<INDEX_TYPE, VALUE_TYPE, SHARED_MEM_SIZE, THREADS>(
			nnzB, rowsB, 
			rowOffsetsB, colIdsA, colIdsB, valuesA, valuesB,
			colIdsC, valuesC, rowOperations, sortColumns,
			rowMaxOperations, startRow, lastRowExcl, resultStartId, 
			resultNnz, minCol, minCol + colRange, startId, lastIdExcl, 
			setSortedBit, rowsPerBlock);
	}
	else
	{
		hashSpGEMMNumericImplementation<INDEX_TYPE, VALUE_TYPE, SHARED_MEM_SIZE, THREADS>(
			nnzA, nnzB, rowsA, rowsB, rowOffsetsA, rowOffsetsB, colIdsA, colIdsB, valuesA, valuesB, 
			colIdsC, valuesC, rowOperations, sortColumns,
			rowMaxOperations, startRow, lastRowExcl,
			resultStartId, resultNnz, minCol, minCol + colRange, 
			startId, lastIdExcl, setSortedBit, rowsPerBlock);
	}
}

// gets information about the work for this block and then decides which method to use -> hash, dense or direct
template <typename INDEX_TYPE, uint32_t MAX_ROWS_PER_BLOCK, class GlobalHashMap, class GlobalRowOffsetMap, uint32_t SHARED_MEM_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
__global__ void 
__launch_bounds__(1024, 2) 
spGEMMCountLauncher(
	const INDEX_TYPE nnzA, const INDEX_TYPE nnzB, const INDEX_TYPE rowsA, const INDEX_TYPE rowsB, const INDEX_TYPE colsB,
	const INDEX_TYPE *__restrict__ rowOffsetsA, const INDEX_TYPE *__restrict__ rowOffsetsB, const INDEX_TYPE *__restrict__ colIdsA, const INDEX_TYPE *__restrict__ colIdsB,
	GlobalHashMap *hashMaps, INDEX_TYPE hashMapCount, GlobalRowOffsetMap *rowOffsetMaps, INDEX_TYPE rowOffsetMapCount,
	INDEX_TYPE *__restrict__ rowOffsetsC, INDEX_TYPE *__restrict__ blockStartRow,
	const INDEX_TYPE *__restrict__ rowOperations,
	const uint32_t numberBlocks, const INDEX_TYPE *__restrict__ rowColMinMax,
	uint32_t *rowMaxOperations, uint32_t minimumDensity, INDEX_TYPE *maxNnzPerRow, uint32_t rowsPerBlock)
{
	INDEX_TYPE startRow = blockStartRow != nullptr ? blockStartRow[blockIdx.x] : blockIdx.x * rowsPerBlock;
	INDEX_TYPE lastRowExcl;
	if (blockStartRow != nullptr)
	{
		lastRowExcl = blockRangeToNumRows(startRow) + blockRangeToStartRow(startRow);
		startRow = blockRangeToStartRow(startRow);
	}
	else
		lastRowExcl = blockIdx.x + 1 < numberBlocks ? (blockStartRow != nullptr ? blockStartRow[blockIdx.x + 1] : (blockIdx.x + 1) * rowsPerBlock) : rowsA;
	lastRowExcl = min(lastRowExcl, rowsA);

	const INDEX_TYPE startId = rowOffsetsA[startRow];
	const INDEX_TYPE lastIdExcl = lastRowExcl < rowsA ? rowOffsetsA[lastRowExcl] : nnzA;
	const INDEX_TYPE maxNnz = min(colsB, rowMaxOperations != nullptr ? rowMaxOperations[startRow] : colsB);
	const INDEX_TYPE minCol = rowColMinMax != nullptr ? rowColMinMaxtoMinCol(rowColMinMax[startRow]) : 0;
	const INDEX_TYPE rowColSize = rowColMinMax != nullptr ? rowColMinMaxtoRowLength(rowColMinMax[startRow]) : spECK::numeric_limits<INDEX_TYPE>::max();
	const INDEX_TYPE colRange = min(colsB - minCol, rowColSize);

	const int hashMapSize = SHARED_MEM_SIZE / sizeof(INDEX_TYPE) - THREADS;

	if (lastIdExcl - startId == 1 && lastRowExcl - startRow == 1)
	{
		directSpGEMMCountImplementation<INDEX_TYPE, THREADS>(
			rowOffsetsB, colIdsA, colIdsB, rowOffsetsC, startRow, rowsB, nnzB, startId, lastIdExcl, maxNnzPerRow);
	}
	else if (lastRowExcl - startRow == 1 && (maxNnz >= hashMapSize || maxNnz * 100 > colRange * minimumDensity) && (lastIdExcl - startId) < (SHARED_MEM_SIZE / 2 / sizeof(INDEX_TYPE)))
	{
		denseSpGEMMCountImplementation<INDEX_TYPE, SHARED_MEM_SIZE, SUPPORT_GLOBAL, THREADS>(
			nnzB, rowsB, colsB, rowOffsetsB, colIdsA, colIdsB, rowOffsetMaps, rowOffsetMapCount, rowOffsetsC,
			rowOperations, rowMaxOperations, maxNnzPerRow, startId, lastIdExcl, startRow, minCol, minCol + colRange, rowsPerBlock);
	} else if (lastRowExcl - startRow == 1 && !SUPPORT_GLOBAL) {
		hashSpGEMMCountSingleRowImplementation<INDEX_TYPE, SHARED_MEM_SIZE, THREADS>(
			rowOffsetsB, colIdsA, colIdsB,
			rowOffsetsC, rowOperations,
			rowMaxOperations, maxNnzPerRow,
			startId, lastIdExcl, startRow,
			minCol, minCol + colRange);
	} 
	else
	{
		hashSpGEMMCountImplementation<INDEX_TYPE, MAX_ROWS_PER_BLOCK, GlobalHashMap, SHARED_MEM_SIZE, SUPPORT_GLOBAL, THREADS>(
			nnzA, nnzB, rowsA, rowsB, rowOffsetsA, rowOffsetsB, colIdsA, colIdsB, hashMaps, hashMapCount, rowOffsetsC, rowOperations,
			rowMaxOperations, maxNnzPerRow, startId, lastIdExcl, startRow, lastRowExcl, minCol, minCol + colRange, rowsPerBlock);
	}
}

// sorts the rows using cub block sort. also removes the row information and the sorted-bit for pre-sorted rows
template <typename INDEX_TYPE, typename VALUE_TYPE, uint32_t THREADS, uint32_t entriesPerBlock>
__global__ void hashSpGEMMSortingKernel(dCSRNoDealloc<VALUE_TYPE> matC, INDEX_TYPE *blockStartRow, uint32_t numberBlocks, bool bitShiftNumRows)
{
	const uint16_t elementsPerThread = (entriesPerBlock + THREADS - 1) / THREADS;
	typedef cub::BlockRadixSort<INDEX_TYPE, THREADS, elementsPerThread, VALUE_TYPE> BlockRadixSort;
	typedef HashMap<INDEX_TYPE, VALUE_TYPE> Map;

	__shared__ typename BlockRadixSort::TempStorage tempStorage;

	INDEX_TYPE threadIndices[elementsPerThread];
	VALUE_TYPE threadValues[elementsPerThread];
	INDEX_TYPE startRow = blockStartRow != nullptr ? blockStartRow[blockIdx.x] : blockIdx.x;

	INDEX_TYPE lastRowExcl;
	if (bitShiftNumRows && blockStartRow != nullptr)
	{
		lastRowExcl = blockRangeToNumRows(startRow) + blockRangeToStartRow(startRow);
		startRow = blockRangeToStartRow(startRow);
	}
	else
		lastRowExcl = blockIdx.x + 1 < numberBlocks ? (blockStartRow != nullptr ? blockStartRow[blockIdx.x + 1] : (blockIdx.x + 1)) : matC.rows;

	const INDEX_TYPE startId = matC.row_offsets[startRow];
	const INDEX_TYPE lastIdExcl = lastRowExcl < matC.rows ? matC.row_offsets[lastRowExcl] : matC.nnz;

	INDEX_TYPE nnz = lastIdExcl - startId;

	if (__syncthreads_or(isRowSorted(matC.col_ids[startId])))
	{
		if (threadIdx.x == 0)
			removeSortedMark(matC.col_ids[startId]);
		return;
	}

	if (nnz >= entriesPerBlock || nnz < 500)
		return;
	
	// get part of hash in local memory / registers
#pragma unroll
	for (int i = 0; i < elementsPerThread; ++i)
	{
		uint32_t mapIndex = startId + i + threadIdx.x * elementsPerThread;

		if (mapIndex < lastIdExcl)
		{
			threadIndices[i] = matC.col_ids[mapIndex];
			threadValues[i] = matC.data[mapIndex];
		} else
		{
			threadIndices[i] = spECK::numeric_limits<INDEX_TYPE>::max();
		}
	}

	BlockRadixSort(tempStorage).Sort(threadIndices, threadValues);

#pragma unroll
	for (int i = 0; i < elementsPerThread; ++i)
	{
		if (threadIndices[i] == Map::UNUSED())
			return;

		uint32_t mapIndex = threadIdx.x * elementsPerThread + i + startId;

		if (mapIndex >= lastIdExcl)
			return;

		matC.col_ids[mapIndex] = Map::idToCol(threadIndices[i]);
		matC.data[mapIndex] = threadValues[i];
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE, uint32_t THREADS, uint32_t entriesPerBlock>
void spECKKernels::h_HashSpGEMMSorting(dCSRNoDealloc<VALUE_TYPE> matC, INDEX_TYPE *blockStartRow, uint32_t numberBlocks, bool bitShiftNumRows)
{
	hashSpGEMMSortingKernel<INDEX_TYPE, VALUE_TYPE, THREADS, entriesPerBlock> << <gridDim, blockDim, sharedMem, stream >> > (matC, blockStartRow, numberBlocks, bitShiftNumRows);
}

template <typename HashMap, typename INDEX_TYPE>
__global__ void initializeGlobalMapsNoVal(HashMap *maps, int count, INDEX_TYPE *ids, size_t elementsPerMap, uint32_t maxRowsPerBlock)
{
	if (threadIdx.x == 0)
	{
		// offset = elementsPerMap + 32 (maxRowsPerBlock) + 1 (occupancy)
		maps[blockIdx.x].ids = ids + ((elementsPerMap + maxRowsPerBlock + 1) * blockIdx.x);
		maps[blockIdx.x].occupancyPerRow = &maps[blockIdx.x].ids[elementsPerMap];
		maps[blockIdx.x].occupancy = &maps[blockIdx.x].occupancyPerRow[maxRowsPerBlock];
		maps[blockIdx.x].capacity = elementsPerMap;
		maps[blockIdx.x].reserved = 0;
	}
	__syncthreads();
	maps[blockIdx.x].init(threadIdx.x == 0);
}

template <typename INDEX_TYPE, typename VALUE_TYPE, class GlobalMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
void spECKKernels::h_HashSpGEMMNumeric(dCSRNoDealloc<VALUE_TYPE> matA, dCSRNoDealloc<VALUE_TYPE> matB, dCSRNoDealloc<VALUE_TYPE> matC, GlobalMap *maps, INDEX_TYPE mapCount,
	INDEX_TYPE *blockStartRow, INDEX_TYPE *rowOperations, Config::SortModes sortColumns, uint32_t numberBlocks, const INDEX_TYPE* rowColMinMax,
	INDEX_TYPE *rowMaxOperations, bool setSortedBit, uint32_t rowsPerBlock)
{
	cudaFuncSetAttribute(hashSpGEMMNumeric<INDEX_TYPE, VALUE_TYPE, GlobalMap, SHARED_HASH_SIZE, SUPPORT_GLOBAL, THREADS>, 
		cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMem);
	hashSpGEMMNumeric<INDEX_TYPE, VALUE_TYPE, GlobalMap, SHARED_HASH_SIZE, SUPPORT_GLOBAL, THREADS><<<gridDim, blockDim, sharedMem, stream>>>(
		matA.nnz, matB.nnz, matA.rows, matB.rows, matB.cols, matA.row_offsets, matB.row_offsets, matA.col_ids, matB.col_ids, matA.data, matB.data,
		maps, mapCount,
		matC.col_ids, matC.data, matC.row_offsets, matC.nnz, blockStartRow, rowOperations, sortColumns, numberBlocks, rowColMinMax,
		rowMaxOperations, setSortedBit, rowsPerBlock);
}

template <typename INDEX_TYPE, typename VALUE_TYPE, unsigned MAX_ROWS_PER_BLOCK, class GlobalMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
void spECKKernels::h_HashSpGEMMCount(dCSRNoDealloc<VALUE_TYPE> matA, dCSRNoDealloc<VALUE_TYPE> matB,
	GlobalMap *hashMaps, INDEX_TYPE hashMapCount,
	INDEX_TYPE *matCNnzRow, INDEX_TYPE *rowOperations, INDEX_TYPE *blockStartRow,
	uint32_t numberBlocks, const INDEX_TYPE *rowColMinMax, INDEX_TYPE *rowMaxOperations, uint32_t *maxNnzPerRow, uint32_t rowsPerBlock)
{
	cudaFuncSetAttribute(hashSpGEMMCount<INDEX_TYPE, MAX_ROWS_PER_BLOCK, GlobalMap, SHARED_HASH_SIZE, SUPPORT_GLOBAL, THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMem);
	hashSpGEMMCount<INDEX_TYPE, MAX_ROWS_PER_BLOCK, GlobalMap, SHARED_HASH_SIZE, SUPPORT_GLOBAL, THREADS> << <gridDim, blockDim, sharedMem, stream >> > (
		matA.nnz, matB.nnz, matA.rows, matB.rows, matB.cols, matA.row_offsets, matB.row_offsets, matA.col_ids, matB.col_ids,
		hashMaps, hashMapCount, matCNnzRow, rowOperations, blockStartRow, numberBlocks, rowColMinMax, 
		rowMaxOperations, maxNnzPerRow, rowsPerBlock);
}

template <typename INDEX_TYPE, typename VALUE_TYPE, unsigned MAX_ROWS_PER_BLOCK, class GlobalMap, class GlobalRowOffsetsMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
void spECKKernels::h_SpGEMMCountLauncher(dCSRNoDealloc<VALUE_TYPE> matA, dCSRNoDealloc<VALUE_TYPE> matB,
	GlobalMap *hashMaps, INDEX_TYPE hashMapCount, GlobalRowOffsetsMap *rowOffsetMaps, INDEX_TYPE rowOffsetMapsCount,
	INDEX_TYPE *matCNnzRow, INDEX_TYPE* rowOperations, INDEX_TYPE *blockStartRow,
	uint32_t numberBlocks, const INDEX_TYPE* rowColMinMax, 
	INDEX_TYPE *rowMaxOperations, uint32_t minimumDensity, INDEX_TYPE *maxNnzPerRow, uint32_t rowsPerBlock)
{
	cudaFuncSetAttribute(spGEMMCountLauncher<INDEX_TYPE, MAX_ROWS_PER_BLOCK, GlobalMap, GlobalRowOffsetsMap, SHARED_HASH_SIZE, SUPPORT_GLOBAL, THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMem);
	spGEMMCountLauncher<INDEX_TYPE, MAX_ROWS_PER_BLOCK, GlobalMap, GlobalRowOffsetsMap, SHARED_HASH_SIZE, SUPPORT_GLOBAL, THREADS> << <gridDim, blockDim, sharedMem, stream >> > (
		matA.nnz, matB.nnz, matA.rows, matB.rows, matB.cols, matA.row_offsets, matB.row_offsets, matA.col_ids, matB.col_ids,
		hashMaps, hashMapCount, rowOffsetMaps, rowOffsetMapsCount, matCNnzRow, blockStartRow, rowOperations, numberBlocks, rowColMinMax,
		rowMaxOperations, minimumDensity, maxNnzPerRow, rowsPerBlock);
}

template <typename INDEX_TYPE, typename VALUE_TYPE, class GlobalMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
void spECKKernels::h_DenseSpGEMMNumeric(dCSRNoDealloc<VALUE_TYPE> matA, dCSRNoDealloc<VALUE_TYPE> matB, dCSRNoDealloc<VALUE_TYPE> matC, GlobalMap *maps, INDEX_TYPE mapCount,
	INDEX_TYPE *blockStartRow, INDEX_TYPE *rowOperations, uint32_t numberBlocks, const INDEX_TYPE* rowColMinMax,
	INDEX_TYPE *rowMaxOperations, bool setSortedBit, uint32_t rowsPerBlock)
{
	cudaFuncSetAttribute(denseSpGEMMNumeric<INDEX_TYPE, VALUE_TYPE, GlobalMap, SHARED_HASH_SIZE, SUPPORT_GLOBAL, THREADS>, 
		cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMem);
	denseSpGEMMNumeric<INDEX_TYPE, VALUE_TYPE, GlobalMap, SHARED_HASH_SIZE, SUPPORT_GLOBAL, THREADS><<<gridDim, blockDim, sharedMem, stream>>>(
		matA.nnz, matB.nnz, matA.rows, matB.rows, matB.cols, matA.row_offsets, matB.row_offsets, matA.col_ids, matB.col_ids, matA.data, matB.data,
		maps, mapCount,
		matC.col_ids, matC.data, matC.row_offsets, matC.nnz,
		blockStartRow, rowOperations, numberBlocks, rowColMinMax,
		rowMaxOperations, setSortedBit, rowsPerBlock);
}

template <typename INDEX_TYPE, typename VALUE_TYPE, class GlobalMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
void spECKKernels::h_DenseSpGEMMCount(dCSRNoDealloc<VALUE_TYPE> matA, dCSRNoDealloc<VALUE_TYPE> matB, GlobalMap *maps, INDEX_TYPE mapCount,
	INDEX_TYPE *rowOffsetsC, INDEX_TYPE *blockStartRow, INDEX_TYPE *rowOperations, uint32_t numberBlocks, const INDEX_TYPE* rowColMinMax,
	INDEX_TYPE *rowMaxOperations, uint32_t *maxNnzPerRow, uint32_t rowsPerBlock)
{
	cudaFuncSetAttribute(denseSpGEMMCount<INDEX_TYPE, GlobalMap, SHARED_HASH_SIZE, SUPPORT_GLOBAL, THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMem);
	denseSpGEMMCount<INDEX_TYPE, GlobalMap, SHARED_HASH_SIZE, SUPPORT_GLOBAL, THREADS><<<gridDim, blockDim, sharedMem, stream>>>(
		matA.nnz, matB.nnz, matA.rows, matB.rows, matB.cols, matA.row_offsets, matB.row_offsets, matA.col_ids, matB.col_ids,
		maps, mapCount, rowOffsetsC, blockStartRow, rowOperations, numberBlocks, rowColMinMax,
		rowMaxOperations, maxNnzPerRow, 
		rowsPerBlock);
}


template <typename INDEX_TYPE, typename VALUE_TYPE, class GlobalHashMap, class GlobalRowOffsetMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
void spECKKernels::h_SpGEMMNumericLauncher(dCSRNoDealloc<VALUE_TYPE> matA, dCSRNoDealloc<VALUE_TYPE> matB, dCSRNoDealloc<VALUE_TYPE> matC,
	GlobalHashMap *hashMaps, INDEX_TYPE hashMapCount, GlobalRowOffsetMap *rowOffsetMaps, INDEX_TYPE rowOffsetMapCount,
	INDEX_TYPE *blockStartRow, INDEX_TYPE *rowOperations, Config::SortModes sortColumns, uint32_t numberBlocks, const INDEX_TYPE* rowColMinMax,
	INDEX_TYPE *rowMaxOperations, uint32_t minimumDensity, bool setSortedBit, uint32_t rowsPerBlock)
{
	cudaFuncSetAttribute(spGEMMNumericLauncher<INDEX_TYPE, VALUE_TYPE, GlobalHashMap, GlobalRowOffsetMap, SHARED_HASH_SIZE, SUPPORT_GLOBAL, THREADS>, 
		cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMem);
	spGEMMNumericLauncher<INDEX_TYPE, VALUE_TYPE, GlobalHashMap, GlobalRowOffsetMap, SHARED_HASH_SIZE, SUPPORT_GLOBAL, THREADS><<<gridDim, blockDim, sharedMem, stream>>>(
		matA.nnz, matB.nnz, matC.nnz, matA.rows, matB.rows, matB.cols, matA.row_offsets, matB.row_offsets, matA.col_ids, matB.col_ids, matA.data, matB.data,
		hashMaps, hashMapCount, rowOffsetMaps, rowOffsetMapCount,
		matC.col_ids, matC.data, matC.row_offsets, blockStartRow, rowOperations, sortColumns, numberBlocks, rowColMinMax,
		rowMaxOperations, minimumDensity, setSortedBit, rowsPerBlock);
}

template <typename Map, typename INDEX_TYPE>
void spECKKernels::h_InitializeGlobalMapsNoVal(Map *maps, int count, INDEX_TYPE *ids, size_t elementsPerMap, uint32_t maxRowsPerBlock)
{
	initializeGlobalMapsNoVal << <gridDim, blockDim >> > (maps, count, ids, elementsPerMap, maxRowsPerBlock);
}