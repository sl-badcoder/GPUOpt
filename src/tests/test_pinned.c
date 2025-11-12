//------------------------------------------------------------------------------------------------------------
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
//------------------------------------------------------------------------------------------------------------
#include "core/helper.h"
#include "gpu/hybrid_gpu.h"
#include "gpu/bitonic_sort.h"
#include "cpu/bitonic.h"
//------------------------------------------------------------------------------------------------------------

void run_gpu(uint32_t* arr, int N){
    hybrid_sort(arr, N, 4096);

}

void test_mapped(){

    cudaSetDeviceFlags(cudaDeviceMapHost);

    srand((unsigned int)time(NULL));

    // trash cache to get good results

    uint32_t *data = create_random_data_u32_mapped(N, UINT32_MAX);
    if(!data) {
        return -1;
    }

    double start = getCurTime();

    double end = getCurTime();

    if(!is_sorted_u32(data, N)){
        printf("Array not sorted!");
    }else{
        printf("GPU Sort in %0.8f -> %0.8f per elem!\n", end - start, ((end - start) / (double) N));
    }

    cudaFreeHost(data);

}


