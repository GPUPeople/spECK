#pragma once

#include <cuda_runtime.h>
#include <type_traits>
#include "GPU/limits.cuh"
#include <device_launch_parameters.h>

__device__ __forceinline__ uint32_t toHashEntry(uint32_t row, uint32_t col)
{
    return (row << 27) + col;
}

__device__ __forceinline__ uint32_t hashEntryToColumn(uint32_t hashEntry)
{
    return hashEntry & 0x7FFFFFF;
}

__device__ __forceinline__ uint32_t hashEntryToRow(uint32_t hashEntry)
{
    return hashEntry >> 27;
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
struct HashMap
{
  public:
    // no default values or else union does not work
    INDEX_TYPE *ids;
    VALUE_TYPE *values;
    uint32_t capacity;

    __device__ __forceinline__ static INDEX_TYPE UNUSED() { return spECK::numeric_limits<INDEX_TYPE>::max(); }

    __device__ void init()
    {
        for (int i = threadIdx.x; i < capacity; i += blockDim.x)
        {
            ids[i] = UNUSED();
            values[i] = VALUE_TYPE(0);
        }
    }

    __device__ __forceinline__ INDEX_TYPE indexOf(INDEX_TYPE id)
    {
        INDEX_TYPE hashed_id = currentHash(id);
        INDEX_TYPE map_id = hashed_id % getSize();
        do
        {
            auto entry_id = ids[map_id];
            if (entry_id == id)
            {
                return map_id;
            }

            if (entry_id == UNUSED())
            {
                auto old_id = atomicCAS(ids + map_id, UNUSED(), id);
                if (old_id == UNUSED() || old_id == id)
                {
                    return map_id;
                }
            }

            map_id = (map_id + 1) % getSize();
        } while (true);
    }

    __device__ __forceinline__ VALUE_TYPE &operator[](INDEX_TYPE id)
    {
        return values[indexOf(id)];
    }

    __device__ __forceinline__ INDEX_TYPE coordToId(INDEX_TYPE rowA, INDEX_TYPE colB)
    {
        return toHashEntry(rowA, colB);
    }

    __device__ __forceinline__ static INDEX_TYPE idToRow(INDEX_TYPE id) { return hashEntryToRow(id); }

    __device__ __forceinline__ static INDEX_TYPE idToCol(INDEX_TYPE id) { return hashEntryToColumn(id); }

    __device__ __forceinline__ VALUE_TYPE &at(INDEX_TYPE rowA, INDEX_TYPE colB)
    {
        return this->operator[](coordToId(rowA, colB));
    }

    __device__ __forceinline__ void atomic_add_direct(INDEX_TYPE rowA, INDEX_TYPE colB, VALUE_TYPE val)
    {
        atomicAdd_block(&values[colB], val);
        ids[colB] = coordToId(rowA, colB);
    }

    __device__ __forceinline__ void atomic_add_direct(INDEX_TYPE colB, VALUE_TYPE val)
    {
        atomicAdd_block(&values[colB], val);
        ids[colB] = colB;
    }

    __device__ __forceinline__ void atomic_add(INDEX_TYPE rowA, INDEX_TYPE colB, VALUE_TYPE val)
    {
        atomic_add(coordToId(rowA, colB), val);
    }

    __device__ __forceinline__ void atomic_add(INDEX_TYPE id, VALUE_TYPE val)
    {
        atomicAdd_block(values + this->indexOf(id), val);
    }

    __device__ __forceinline__ uint32_t getSize() const { return capacity; }
};

template <class HashMap>
__device__ __forceinline__ HashMap *reserveMap(HashMap *maps, uint32_t count)
{
    uint32_t index = blockIdx.x % count;

    while (true)
    {
        if (atomicCAS(&maps[index].reserved, 0, 1) == 0)
        {
            return &maps[index];
        }
        index = (index + 1) % count;
    }
}

template <class HashMap>
__device__ __forceinline__ void freeMap(HashMap *map)
{
    if (map == nullptr)
        return;
    map->reserved = 0;
    map = nullptr;
}

template <typename INDEX_TYPE, size_t MAX_ROW_COUNT>
struct HashMapNoValue
{
  private:
    uint32_t limit;

  public:
    __device__ INDEX_TYPE UNUSED() const { return spECK::numeric_limits<INDEX_TYPE>::max(); }
    INDEX_TYPE *ids;
    INDEX_TYPE *occupancyPerRow;
    INDEX_TYPE *occupancy;

    // no default values or else union does not work
    int reserved;
    uint32_t capacity;

    __device__ void init(bool mainThread)
    {
        for (int i = threadIdx.x; i < capacity; i += blockDim.x)
            ids[i] = UNUSED();

        for (int i = threadIdx.x; i < MAX_ROW_COUNT; i += blockDim.x)
            occupancyPerRow[i] = 0;

        if (mainThread)
        {
            *occupancy = 0;
            limit = capacity;
        }
    }

    __device__ __forceinline__ void operator[](INDEX_TYPE id)
    {
        INDEX_TYPE hashed_id = currentHash(id);
        INDEX_TYPE map_id = hashed_id % getSize();

        do
        {
            auto entry = ids[map_id];
            if (entry == id)
                return;

            if (entry == UNUSED())
            {
                auto old_id = atomicCAS(ids + map_id, UNUSED(), id);

                if (old_id == UNUSED() || old_id == id)
                {
                    if (old_id == UNUSED())
                    {
                        atomicAdd_block(occupancy, 1);
                        atomicAdd_block(&occupancyPerRow[idToRow(id)], 1);
                    }
                    return;
                }
            }

            map_id = (map_id + 1) % getSize();
        } while (true);
    }

    __device__ __forceinline__ void limitSize(uint32_t limit)
    {
        this->limit = min(limit, capacity);
    }

    __device__ __forceinline__ INDEX_TYPE coordToId(INDEX_TYPE rowA, INDEX_TYPE colB)
    {
        return toHashEntry(rowA, colB);
    }

    __device__ __forceinline__ static INDEX_TYPE idToRow(INDEX_TYPE id) { return hashEntryToRow(id); }

    __device__ __forceinline__ static INDEX_TYPE idToCol(INDEX_TYPE id) { return hashEntryToColumn(id); }

    __device__ __forceinline__ void at(INDEX_TYPE rowA, INDEX_TYPE colB)
    {
        this->operator[](coordToId(rowA, colB));
    }

    __device__ __forceinline__ void atDirect(INDEX_TYPE rowA, INDEX_TYPE colB)
    {
        if (ids[colB] != UNUSED())
            return;

        INDEX_TYPE retVal = atomicCAS(&ids[colB], UNUSED(), coordToId(rowA, colB));
        if (retVal == UNUSED())
        {
            atomicAdd_block(occupancy, 1);
            atomicAdd_block(&occupancyPerRow[rowA], 1);
        }
    }

    __device__ __forceinline__ size_t getSize() const { return limit; }
};