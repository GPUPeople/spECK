#pragma once

#include <stdio.h>
#include <cuda.h>
#include <cupti.h>
#include <vector>

// Structure to hold data collected by callback
typedef struct RuntimeApiTrace_st {
  const char *functionName;
  uint64_t startTimestamp;
  uint64_t endTimestamp;
  size_t memcpy_bytes;
  enum cudaMemcpyKind memcpy_kind;
  size_t currentMemoryUsage;
} RuntimeApiTrace_t;

enum launchOrder{ MEMCPY_H2D1, MEMCPY_H2D2, MEMCPY_D2H, KERNEL, THREAD_SYNC, LAUNCH_LAST};

class CuProfiler {
private:
    std::vector<RuntimeApiTrace_st> records;
    CUpti_SubscriberHandle subscriber;
    unsigned long long int startMem;
    bool initialized;

public:
    RuntimeApiTrace_t trace[LAUNCH_LAST];

    CuProfiler() {
        initialized = false;
    }


    // void CUPTIAPI getTimestampCallback(void *userdata, CUpti_CallbackDomain domain,
    //                     CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo);

    size_t getPeakMemoryUsage() {
        if (records.size() == 0)
            return -1;

        size_t max = 0;
        for (auto& entry : records) {
            if (entry.currentMemoryUsage > max)
                max = entry.currentMemoryUsage;
        }
        return max;
    }

    static const char *memcpyKindStr(enum cudaMemcpyKind kind)
    {
        switch (kind) {
        case cudaMemcpyHostToDevice:
            return "HostToDevice";
        case cudaMemcpyDeviceToHost:
            return "DeviceToHost";
        default:
            break;
        }

        return "<unknown>";
    }

    void displayTimestamps();


    static void cleanUp(int *h_A, int *h_B, int *h_C, int *d_A, int *d_B, int *d_C)
    {
        if (d_A)
            cudaFree(d_A);
        if (d_B)
            cudaFree(d_B);
        if (d_C)
            cudaFree(d_C);

        // Free host memory
        if (h_A)
            free(h_A);
        if (h_B)
            free(h_B);
        if (h_C)
            free(h_C);
    }

    void initialize(bool subtractCurrentMem = true);
    void finalize();
};