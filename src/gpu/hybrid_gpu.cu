//------------------------------------------------------------------------------------------------------------
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
//------------------------------------------------------------------------------------------------------------
#include "gpu/bitonic_sort.h"
#include "gpu/hybrid_gpu.h"
#include "cpu/bitonic_simd_merge.h"
#include "core/helper.h"
//------------------------------------------------------------------------------------------------------------
extern "C" void hybrid_sort(uint32_t *arr, int N, int K_CPU_MAX) {
    int K = (K_CPU_MAX);
    /* printf("K: %d", K);
    printf("N: %d", N);
    bool use_cpu = false;
    if(K_CPU_MAX < 67108864)use_cpu = true;
    //------------------------------------------------------------------------------------------------------------
    clock_t start, end;
    double cpu_time_used;
    //------------------------------------------------------------------------------------------------------------
    struct timespec a, b; 
    clock_gettime(CLOCK_MONOTONIC,&a);*/
    //------------------------------------------------------------------------------------------------------------
    simd_mergesort_uint32_k(arr, N, K);
    //clock_gettime(CLOCK_MONOTONIC,&b);
    //double sec = (b.tv_sec-a.tv_sec) + (b.tv_nsec-a.tv_nsec)/1e9;
    //printf("CPU time used sort: %.6f seconds\n", sec);
    //------------------------------------------------------------------------------------------------------------
    //clock_gettime(CLOCK_MONOTONIC,&a);
    //if (src != arr) memcpy(arr, src, N * sizeof(uint32_t));
    make_alternating_runs(arr, N, K);
    //clock_gettime(CLOCK_MONOTONIC,&b);
    //sec = (b.tv_sec-a.tv_sec) + (b.tv_nsec-a.tv_nsec)/1e9;
    //printf("CPU time used alternating: %.6f seconds\n", sec);
    //------------------------------------------------------------------------------------------------------------
    /**cudaEvent_t start_shared, stop_shared;
    float ms = 0.0f;
    cudaEventCreate(&start_shared);
    cudaEventCreate(&stop_shared);
    cudaEventRecord(start_shared);**/
    //cudaMemcpy(d_data, arr, N*4, cudaMemcpyHostToDevice);
    gpu_bitonic_sort_uint32_k(arr, N, K, false);
    //cudaMemcpy(arr, d_data, N*4, cudaMemcpyDeviceToHost);
    /*cudaDeviceSynchronize();
    cudaEventRecord(stop_shared);
    cudaEventSynchronize(stop_shared);
    //------------------------------------------------------------------------------------------------------------ 
    cudaEventElapsedTime(&ms, start_shared, stop_shared);
    printf("GPU Time: %.3f s\n", (ms / 1000));
    cudaEventDestroy(start_shared);
    cudaEventDestroy(stop_shared);*/
    //------------------------------------------------------------------------------------------------------------
}