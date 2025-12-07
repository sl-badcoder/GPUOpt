//------------------------------------------------------------------------------------------------------------
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
//------------------------------------------------------------------------------------------------------------
#include "cpu/bitonic.h"
#include "cpu/bitonic_cellsort.h"
#include "gpu/hybrid_gpu.h"
#include "core/helper.h"
//------------------------------------------------------------------------------------------------------------
// this is the main function where all programms are called
//------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
    size_t N;
    if(argc < 2) {
        printf("./sort [NUM_OF_ELEM]\n");
        return 0;
    }
    N = 2 << atol(argv[1]);
    if(N % 2 != 0) {
        printf("N must be a power of 2\n");
        return 0;
    }
    string m = argv[2];
    if(m != "unified" || m != "pinned"){
        printf("it must be specified to either use unified or pinned memory (via mapped/pinned)\n")
        return 0;
    }
    //------------------------------------------------------------------------------------------------------------
    // Init Rand
    srand((unsigned int)time(NULL));
    if(m == "mapped"){
        uint32_t *data = create_random_data_u32_unified(N, UINT32_MAX);
        if(!data) {
            return -1;
        }
        double start = getCurTime();
        hybrid_sort_unified(data, N,16);
        double end = getCurTime();
        if(!is_sorted_u32(data, N)) {
            printf("Array not sorted\n");
        } else {
            printf("GPU Sort in %0.8f -> %0.8f per elem!\n", end - start, ((end - start) / (double) N));
        }
        //------------------------------------------------------------------------------------------------------------
        cudaFree(data);
        //------------------------------------------------------------------------------------------------------------
    }else{
        uint32_t * data = create_random_data_u32_pinned(N, UINT32_MAX);
        if(!data){
            return -1;
        }
        double start = getCurTime();
        hybrid_sort_unified(data, N,16);
        double end = getCurTime();
        if(!is_sorted_u32(data, N)) {
            printf("Array not sorted\n");
        } else {
            printf("GPU Sort in %0.8f -> %0.8f per elem!\n", end - start, ((end - start) / (double) N));
        }
        //------------------------------------------------------------------------------------------------------------
        cudaFreeHost(data);
    }
    //------------------------------------------------------------------------------------------------------------
    return 0;
}
//------------------------------------------------------------------------------------------------------------