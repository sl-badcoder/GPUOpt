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
// this is the main function where all prgramms are called
#define CHECK_CUDA(call) do {                                         \
  cudaError_t _e = (call);                                            \
  if (_e != cudaSuccess) {                                            \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
            cudaGetErrorString(_e));                                  \
    exit(1);                                                          \
  }                                                                   \
} while (0)
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
    //------------------------------------------------------------------------------------------------------------
    // Init Rand
    srand((unsigned int)time(NULL));
    //------------------------------------------------------------------------------------------------------------
    //cudaSetDeviceFlags(cudaDeviceMapHost);
    CHECK_CUDA(cudaGetLastError());
    //------------------------------------------------------------------------------------------------------------
    {

        // UINT SORT
        uint32_t *data = create_random_data_u32(N, UINT32_MAX);
        if(!data) {
            return -1;
        }
        //------------------------------------------------------------------------------------------------------------
        double start = getCurTime();
        simd_mergesort_uint32(data, N);
        double end = getCurTime();
        if(!is_sorted_u32(data, N)) {
            printf("Array not sorted\n");
        } else {
            printf("Bitonic Sort in %0.8f -> %0.8f per elem!\n", end - start, ((end - start) / (double) N));
        }
        free(data);
    }
    {
        // UINT SORT
        //for(int K =16; K < 4096; K<<=1){
            uint32_t *data = create_random_data_u32_pinned(N, UINT32_MAX);
            if(!data) {
                return -1;
            }
            //------------------------------------------------------------------------------------------------------------
            // Hybrid Sort
            //------------------------------------------------------------------------------------------------------------
            double start = getCurTime();
            //hybrid_sort(data, N,16);
            gpu_bitonic_sort_uint32(data, N);
            double end = getCurTime();
            if(!is_sorted_u32(data, N)) {
                printf("GPU Sort in %0.8f -> %0.8f per elem!\n", end - start, ((end - start) / (double) N));
                printf("Array not sorted\n");
            } else {
                printf("K: %d", 16);
                printf("GPU Sort in %0.8f -> %0.8f per elem!\n", end - start, ((end - start) / (double) N));
            }
            //------------------------------------------------------------------------------------------------------------
            cudaFreeHost(data);
        //}
        //------------------------------------------------------------------------------------------------------------
    }
    //------------------------------------------------------------------------------------------------------------
    return 0;
}
//------------------------------------------------------------------------------------------------------------