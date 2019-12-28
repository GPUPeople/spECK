#include "GPU/profiler.cuh"

#define CHECK_CU_ERROR(err, cufunc)                                     \
  if (err != CUDA_SUCCESS)                                              \
    {                                                                   \
      printf ("%s:%d: error %d for CUDA Driver API function '%s'\n",    \
              __FILE__, __LINE__, err, cufunc);                         \
      exit(-1);                                                         \
    }

#define CHECK_CUPTI_ERROR(err, cuptifunc)                               \
  if (err != CUPTI_SUCCESS)                                             \
    {                                                                   \
      const char *errstr;                                               \
      cuptiGetResultString(err, &errstr);                               \
      printf ("%s:%d:Error %s for CUPTI API function '%s'.\n",          \
              __FILE__, __LINE__, errstr, cuptifunc);                   \
      exit(-1);                                                         \
    }


// void CUPTIAPI getTimestampCallback(void *userdata, CUpti_CallbackDomain domain,
//                      CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
// {
//     static int memTransCount = 0;
//     uint64_t startTimestamp;
//     uint64_t endTimestamp;
//     RuntimeApiTrace_t *traceData = (RuntimeApiTrace_t*)userdata;
//     CUptiResult cuptiErr;
        
//     // Data is collected only for the following API
//     if ((cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) ||
//         (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) ||
//         (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020) || 
//         (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020))  { 
        
//         // Set pointer depending on API
//         if ((cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) ||
//             (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000))
//         {
//         traceData = traceData + KERNEL;
//         }
//         else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020) 
//         traceData = traceData + THREAD_SYNC;
//         else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020)
//         traceData = traceData + MEMCPY_H2D1 + memTransCount;
//         size_t freeMem = 0, totalMem = 0;
//         cudaMemGetInfo(&freeMem, &totalMem);
//         traceData->currentMemoryUsage = totalMem - freeMem;
                    
//         if (cbInfo->callbackSite == CUPTI_API_ENTER) {
//             // for a kernel launch report the kernel name, otherwise use the API
//             // function name.
//             if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
//                 cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000)
//             {
//                 traceData->functionName = cbInfo->symbolName;
//             }
//             else {
//                 traceData->functionName = cbInfo->functionName;
//             }

//             // Store parameters passed to cudaMemcpy
//             if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
//                 traceData->memcpy_bytes = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams))->count;
//                 traceData->memcpy_kind = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams))->kind;
//             }
                
//             // Collect timestamp for API start
//             cuptiErr = cuptiDeviceGetTimestamp(cbInfo->context, &startTimestamp);
//             CHECK_CUPTI_ERROR(cuptiErr, "cuptiDeviceGetTimestamp");
                    
//             traceData->startTimestamp = startTimestamp;
//             }

//             if (cbInfo->callbackSite == CUPTI_API_EXIT) {
//             // Collect timestamp for API exit
//             cuptiErr = cuptiDeviceGetTimestamp(cbInfo->context, &endTimestamp);
//             CHECK_CUPTI_ERROR(cuptiErr, "cuptiDeviceGetTimestamp");
                    
//             traceData->endTimestamp = endTimestamp;
            
//             // Advance to the next memory transfer operation
//             if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
//                 memTransCount++;
//             }
//         } 
//     }
// }

void CUPTIAPI getTimestampCallback(void *userdata, CUpti_CallbackDomain domain,
                     CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
{
    if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemGetInfo_v3020)
        return;

    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);

    RuntimeApiTrace_t traceData;
    traceData.functionName = cbInfo->functionName;
    traceData.currentMemoryUsage = totalMem - freeMem;

    auto &records = *((std::vector<RuntimeApiTrace_t> *) userdata);
    records.push_back(traceData);
        

    // printf("current usage=%llu. Entry nr=%llu\n", traceData.currentMemoryUsage, (long long unsigned int) records.size());
}

void CuProfiler::initialize(bool subtractCurrentMem) {
    CUcontext context = 0;
    CUdevice device = 0;
    CUresult cuerr;
    CUptiResult cuptierr;

    startMem = 0;
    if(subtractCurrentMem) {
        size_t freeMem = 0, totalMem = 0;
        cudaMemGetInfo(&freeMem, &totalMem);
        startMem = totalMem - freeMem;
    }
    // RuntimeApiTrace_t trace[LAUNCH_LAST];
        
    // cuerr = cuInit(0);
    // CHECK_CU_ERROR(cuerr, "cuInit");

    // cuerr = cuCtxCreate(&context, 0, device);
    // cuerr = cuCtxGetCurrent(&context);
    // CHECK_CU_ERROR(cuerr, "cuCtxCreate");

    cuptierr = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getTimestampCallback, &records);
    CHECK_CUPTI_ERROR(cuptierr, "cuptiSubscribe");

    cuptierr = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    CHECK_CUPTI_ERROR(cuptierr, "cuptiEnableDomain");

    initialized = true;
}

void CuProfiler::finalize() {
    if (!initialized)
        return;

    for (auto& entry : records) {
        if (entry.currentMemoryUsage < startMem)
            startMem =  entry.currentMemoryUsage;
    }

    for (auto& entry : records) {
        entry.currentMemoryUsage -= startMem;
    }

    // displayTimestamps();

    CUptiResult cuptierr = cuptiUnsubscribe(subscriber);
    CHECK_CUPTI_ERROR(cuptierr, "cuptiUnsubscribe");

    // cleanUp(h_A, h_B, h_C, d_A, d_B, d_C);
    cudaDeviceSynchronize();
}


    
void CuProfiler::displayTimestamps()
{
    for (auto entry : records) {
        printf("MemUsage:%llu\n", entry.currentMemoryUsage);
    }
    // Calculate timestamp of kernel based on timestamp from
    // cudaDeviceSynchronize() call
    // trace[KERNEL].endTimestamp = trace[THREAD_SYNC].endTimestamp;

    // printf("startTimeStamp/gpuTime reported in nano-seconds\n\n");
    // printf("Name\t\tStart Time\t\tGPU Time\tBytes\tKind\tcurrentMemUsage\n");
    // printf("%s\t%llu\t%llu\t\t%llu\t%s\t\%llu\n", trace[MEMCPY_H2D1].functionName,
    //         (unsigned long long)trace[MEMCPY_H2D1].startTimestamp, 
    //         (unsigned long long)trace[MEMCPY_H2D1].endTimestamp - trace[MEMCPY_H2D1].startTimestamp,
    //         (unsigned long long)trace[MEMCPY_H2D1].memcpy_bytes,
    //         memcpyKindStr(trace[MEMCPY_H2D1].memcpy_kind),
    //         (unsigned long long)trace[MEMCPY_H2D1].currentMemoryUsage);
    // printf("%s\t%llu\t%llu\t\t%llu\t%s\t\%llu\n", trace[MEMCPY_H2D2].functionName,
    //         (unsigned long long)trace[MEMCPY_H2D2].startTimestamp,
    //         (unsigned long long)trace[MEMCPY_H2D2].endTimestamp - trace[MEMCPY_H2D2].startTimestamp, 
    //         (unsigned long long)trace[MEMCPY_H2D2].memcpy_bytes,
    //         memcpyKindStr(trace[MEMCPY_H2D2].memcpy_kind),
    //         (unsigned long long)trace[MEMCPY_H2D1].currentMemoryUsage); 
    // printf("%s\t%llu\t%llu\t\tNA\tNA\t\%llu\n", trace[KERNEL].functionName,
    //         (unsigned long long)trace[KERNEL].startTimestamp,
    //         (unsigned long long)trace[KERNEL].endTimestamp - trace[KERNEL].startTimestamp,
    //         (unsigned long long)trace[MEMCPY_H2D1].currentMemoryUsage);
    // printf("%s\t%llu\t%llu\t\t%llu\t%s\t\%llu\n", trace[MEMCPY_D2H].functionName,
    //         (unsigned long long)trace[MEMCPY_D2H].startTimestamp,
    //         (unsigned long long)trace[MEMCPY_D2H].endTimestamp - trace[MEMCPY_D2H].startTimestamp, 
    //         (unsigned long long)trace[MEMCPY_D2H].memcpy_bytes,
    //         memcpyKindStr(trace[MEMCPY_D2H].memcpy_kind),
    //         (unsigned long long)trace[MEMCPY_H2D1].currentMemoryUsage); 
}