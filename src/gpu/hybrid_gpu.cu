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
#include "cost_model/gpu_usage.h"
//------------------------------------------------------------------------------------------------------------
extern "C" void hybrid_sort_v(uint32_t *arr, size_t N, size_t K_CPU_MAX) {
    size_t K = (K_CPU_MAX);
    //------------------------------------------------------------------------------------------------------------
    simd_mergesort_uint32_k(arr, N, K);
    make_alternating_runs(arr, N, K);
    gpu_bitonic_sort_uint32_k(arr, N, K, false);
    CHECK_CUDA(cudaDeviceSynchronize());
    //------------------------------------------------------------------------------------------------------------
}
//------------------------------------------------------------------------------------------------------------
extern "C" void hybrid_sort(uint32_t *arr, size_t N, size_t K_CPU_MAX, char* type) {
    size_t K = (K_CPU_MAX);
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
 
    // check if type is unified or not
    if(strcmp(type, "unified") == 0){
        printf("[STARTING] CUDA KERNELS\n");
        CHECK_CUDA(cudaMemPrefetchAsync(arr, N*sizeof(uint32_t), loc, 0, stream));
        gpu_bitonic_sort_uint32_k_un(arr, N, K);
    }else{
        gpu_bitonic_sort_uint32_k(arr, N, K, false);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    //------------------------------------------------------------------------------------------------------------
}

//------------------------------------------------------------------------------------------------------------
extern "C" void hybrid_sort_huge(uint32_t *arr, size_t N, size_t K_CPU_MAX, char* type) {
    size_t free_memory = getGPUFreeMemory() / 1.3;
    printf("[FREE_MEMORY] %zu\n", free_memory);
    if(N * (size_t)sizeof(uint32_t) > getGPUFreeMemory() / 1.3){
    //if(true){ 
        size_t K = (K_CPU_MAX);
        int device = 0;
        CHECK_CUDA(cudaGetDevice(&device));
        //------------------------------------------------------------------------------------------------------------
        simd_mergesort_uint32_k(arr, N, K); // cpu bitonic
        make_alternating_runs(arr, N, K);

        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        // get device location and type
        cudaMemLocation loc{};
        loc.type = cudaMemLocationTypeDevice;   
        loc.id   = device;  
        // split array in half
        uint32_t* left = arr;
        size_t left_size = N / 2;
        uint32_t* right = arr + left_size;
        size_t right_size = left_size;
        printf("[START] SORTING LEFT HALF\n");
        // sort left half on gpu
        CHECK_CUDA(cudaMemPrefetchAsync(left,  left_size * sizeof(uint32_t), loc, 0, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        gpu_bitonic_sort_uint32_k_un(left, left_size, K);

        CHECK_CUDA(cudaMemPrefetchAsync(right, left_size*sizeof(uint32_t), loc, 0, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        //CHECK_CUDA(cudaDeviceSynchronize());
        // sort right half on gpu
        printf("[START] SORTING RIGHT HALF\n");
        gpu_bitonic_sort_uint32_k_un_b(right, right_size, K, false);
        
        cudaMemLocation host_loc{};
        host_loc.type = cudaMemLocationTypeHost;
        host_loc.id   = cudaCpuDeviceId;  
        
        CHECK_CUDA(cudaMemPrefetchAsync(arr, N * sizeof(uint32_t), host_loc, 0, 0));
        CHECK_CUDA(cudaDeviceSynchronize());
        
        printf("[START] SORTING ON CPU\n");
        simd_bitonic_sort_uint32_k(arr, N, left_size);
        //------------------------------------------------------------------------------------------------------------
    }else hybrid_sort(arr, N, K_CPU_MAX, "unified");
}