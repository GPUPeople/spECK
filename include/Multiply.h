
#pragma once

#include "dCSR.h"
#include "Timings.h"
#include "spECKConfig.h"

// REPLACE THESE VALUES WITH YOUR ACTUAL DEVICE SPECIFICATIONS

static constexpr int spECK_STATIC_MEM_PER_BLOCK {49152};
static constexpr int spECK_DYNAMIC_MEM_PER_BLOCK{49152};

namespace spECK
{
    template <typename DataType, int BLOCKS_PER_SM, int THREADS_PER_BLOCK, int MAX_DYNAMIC_SHARED, int MAX_STATIC_SHARED>
    void MultiplyspECK(const dCSR<DataType> &A, const dCSR<DataType> &B, dCSR<DataType> &matOut, spECKConfig &config, Timings &timings);

    template <typename DataType, int BLOCKS_PER_SM, int THREADS_PER_BLOCK, int MAX_DYNAMIC_SHARED, int MAX_STATIC_SHARED>
    void MultiplyspECKImplementation(const dCSR<DataType> &A, const dCSR<DataType> &B, dCSR<DataType> &matOut, spECKConfig &config, Timings &timings = Timings());
} // namespace spECK
