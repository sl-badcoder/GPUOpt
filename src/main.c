//------------------------------------------------------------------------------------------------------------
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
//------------------------------------------------------------------------------------------------------------
#include "cpu/bitonic.h"
#include "cpu/bitonic_cellsort.h"
#include "cpu/bitonic_simd_merge.h"
#include "gpu/hybrid_gpu.h"
#include "core/helper.h"
//------------------------------------------------------------------------------------------------------------
// this is the main function where all programms are called
//------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
    //------------------------------------------------------------------------------------------------------------
    long long N;
    if(strcmp(argv[2],"gpu") == 0 && argc < 4) {
        printf("./sort [NUM_OF_ELEM] [GPU] [MEMORY_TYPE]\n");
        return 0;
    }else if(argc < 3){
        printf("./sort [NUM_OF_ELEM] [CPU]\n");
        return 0;
    }
    //------------------------------------------------------------------------------------------------------------
    N = 1LL << atol(argv[1]);
    //------------------------------------------------------------------------------------------------------------
    if(N % 2 != 0) {
        printf("N must be a power of 2\n");
        return 0;
    }
    //------------------------------------------------------------------------------------------------------------
    long long bytes = N * (long long) sizeof(uint32_t) / (long long)(1024.0 * 1024.0);
    printf("[INPUT_SIZE]: %lld MiB\n", bytes);
    //------------------------------------------------------------------------------------------------------------
    char* m;
    if(strcmp(argv[2],"gpu") == 0){
        m = argv[3];
        printf("[INPUT]: %s\n", m);
        //------------------------------------------------------------------------------------------------------------
        if(strcmp(m, "unified") != 0 &&  strcmp(m,"pinned") != 0 &&  strcmp(m,"unified_huge") != 0){
            printf("it must be specified to either use unified or pinned memory (via unified/pinned)\n");
            return 0;
        }
        //------------------------------------------------------------------------------------------------------------
    }
    //------------------------------------------------------------------------------------------------------------
    // Init Rand
    srand((unsigned int)time(NULL));
    //------------------------------------------------------------------------------------------------------------
    for(int i=0;i<20;i++){
        warmup_cache();
    }
    //------------------------------------------------------------------------------------------------------------
    if(strcmp(argv[2],"cpu")==0){
        printf("[START] CPU TESTCASE...\n");
        uint32_t *data = create_random_data_u32(N, UINT32_MAX);
        if(!data) {
            return -1;
        }
        double start = getCurTime();
        printf("[IN]: %lld \n", N);
        simd_bitonic_sort_uint32(data, N);
        double end = getCurTime();
        if(!is_sorted_u32(data, N)) {
            printf("Array not sorted\n");
            printf("CPU Sort in %0.8f -> %0.8f per elem!\n", end - start, ((end - start) / (double) N));

        } else {
            printf("CPU Sort in %0.8f -> %0.8f per elem!\n", end - start, ((end - start) / (double) N));
        }
        free(data);
    }else if(strcmp(argv[2],"gpu")==0){
        if(strcmp(m, "unified_huge") == 0){
            printf("[START] UNIFIED_HUGE TESTCASE...\n");
            uint32_t *data = create_random_data_u32_unified(N, UINT32_MAX);
            if(!data) {
                return -1;
            }
            double start = getCurTime();
            printf("[IN]: %lld \n", N);
            hybrid_sort_huge(data, N,16, m);
            double end = getCurTime();
            if(!is_sorted_u32(data, N)) {
                printf("Array not sorted\n");
            } else {
                printf("GPU Sort in %0.8f -> %0.8f per elem!\n", end - start, ((end - start) / (double) N));
            }
            //------------------------------------------------------------------------------------------------------------
            CHECK_CUDA(cudaFree(data));
        //------------------------------------------------------------------------------------------------------------
        }else if(strcmp(m, "unified") == 0){
            printf("[START] UNIFIED TESTCASE...\n");
            uint32_t *data = create_random_data_u32_unified(N, UINT32_MAX);
            if(!data) {
                return -1;
            }
            printf("[IN]: %lld \n", N);
            CHECK_CUDA(cudaDeviceSynchronize());
            double start = getCurTime();
            hybrid_sort(data, N,16, m);
            CHECK_CUDA(cudaDeviceSynchronize());

            if(!is_sorted_u32(data, N)) {
                double end = getCurTime();
                printf("Array not sorted\n");
            } else {
                double end = getCurTime();
                printf("GPU Sort in %0.8f -> %0.8f per elem!\n", end - start, ((end - start) / (double) N));
            }
            //------------------------------------------------------------------------------------------------------------
            CHECK_CUDA(cudaFree(data));
        //------------------------------------------------------------------------------------------------------------
        }else {
            printf("[START] PINNED TESTCASE...\n");
            uint32_t * data = create_random_data_u32_pinned(N, UINT32_MAX);
            if(!data){
                return -1;
            }
            double start = getCurTime();
            hybrid_sort(data, N,16, m);
            double end = getCurTime();
            if(!is_sorted_u32(data, N)) {
                printf("Array not sorted\n");
            } else {
                printf("GPU Sort in %0.8f -> %0.8f per elem!\n", end - start, ((end - start) / (double) N));
            }
            //------------------------------------------------------------------------------------------------------------
            CHECK_CUDA(cudaFreeHost(data));
        }
    }
    
    //------------------------------------------------------------------------------------------------------------
    return 0;
}
//------------------------------------------------------------------------------------------------------------