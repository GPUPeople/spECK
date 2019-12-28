#pragma once
#include <vector>
#include "CUDATools/stream.h"
#include "stdio.h"

namespace spECK {
    // get device attributes for best performance and creates cudaStreams
    struct spECKConfig {
        int sm;
        int maxStaticSharedMemoryPerBlock;
        int maxDynamicSharedMemoryPerBlock;
        std::vector<CUstream> streams;
        cudaEvent_t completeStart = 0, completeEnd = 0, individualStart = 0, individualEnd = 0;

        static spECKConfig initialize(int cudaDeviceNumber) {
			spECKConfig config;
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, cudaDeviceNumber);
            config.sm = prop.multiProcessorCount;
            config.maxStaticSharedMemoryPerBlock = prop.sharedMemPerBlock;
            config.maxDynamicSharedMemoryPerBlock = std::max(prop.sharedMemPerBlockOptin, prop.sharedMemPerBlock);

            for (int i = 0; i < 6; i++) {
                config.streams.push_back(0);
                cudaStreamCreate(&config.streams[i]);
            }
            cudaEventCreate(&config.completeStart);
            cudaEventCreate(&config.completeEnd);
            cudaEventCreate(&config.individualStart);
            cudaEventCreate(&config.individualEnd);
            return config;
        }

        void cleanup() {
            for (auto s : streams) {
                cudaStreamDestroy(s);
            }
            cudaEventDestroy(completeStart);
            cudaEventDestroy(completeEnd);
            cudaEventDestroy(individualStart);
            cudaEventDestroy(individualEnd);
            streams.clear();
        }

        ~spECKConfig() {
            // cleanup();
        }

    private:
		spECKConfig() {

        }
    };
}