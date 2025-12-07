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
    //------------------------------------------------------------------------------------------------------------
    simd_mergesort_uint32_k(arr, N, K);
    make_alternating_runs(arr, N, K);
    gpu_bitonic_sort_uint32_k(arr, N, K, false);
    CHECK_CUDA(cudaDeviceSynchronize());
    //------------------------------------------------------------------------------------------------------------
}
//------------------------------------------------------------------------------------------------------------
extern "C" void hybrid_sort(uint32_t *arr, int N, int K_CPU_MAX, string type) {
    int K = (K_CPU_MAX);
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    //------------------------------------------------------------------------------------------------------------
    simd_mergesort_uint32_k(arr, N, K); // cpu bitonic
    make_alternating_runs(arr, N, K);
    
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;   
    loc.id   = device;  
 
    CHECK_CUDA(cudaMemPrefetchAsync(arr, N*sizeof(uint32_t), loc, 0, stream));
    // check if type is unified or not
    if(type == "unified"){
        gpu_bitonic_sort_uint32_k_un(arr, N, K);
    }else{
        gpu_bitonic_sort_uint32_k(arr, N, K, false);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    //------------------------------------------------------------------------------------------------------------
}