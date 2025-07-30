#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

#include "bitonic.h"
#include "bitonic_cellsort.h"
#include "bitonic_simd_merge.h"
#include "bitonic_gpu.h"
#include "helper.h"
#include <CL/cl.h>

// HYBRID APPROACH
// talked about in report
// runs cpu implementation for smaller width and gpu implementation for larger widths
static unsigned query_gpu_compute_units(void)
{
    cl_uint cu = 0; cl_int e;
    cl_platform_id pf; cl_device_id dv;
    e = clGetPlatformIDs(1,&pf,NULL); if(e) return 8;
    e = clGetDeviceIDs(pf,CL_DEVICE_TYPE_GPU,1,&dv,NULL); if(e) return 8;
    clGetDeviceInfo(dv, CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof cu, &cu, NULL);
    return cu ? cu : 8;
}

static void cpu_merge_ladder(uint32_t *bufA,
                             uint32_t *bufB,
                             size_t    n,
                             size_t    *width_io)
{
    size_t width = 4;
    uint32_t *src = bufA, *dst = bufB;

    const unsigned CU = query_gpu_compute_units();

    for (;; width <<= 1)
    {
        size_t runs = (n + (2*width - 1)) / (2*width);
        if (runs <= CU * 16 || width >= n) {
            break;
        }

        simd_merge_pass_uint32(src, dst, width, n);

        uint32_t *swap = src; src = dst; dst = swap;
    }

    if (src != bufA) memcpy(bufA, src, n * sizeof *bufA);
    *width_io = width;
}

int hybrid_tiered_sort_uint32(uint32_t *data, size_t n)
{
    if (n < 8 || (n & (n-1)))
        return -1;

    vector_presort(data, n);
    uint32_t *tmp = aligned_alloc(64, n * sizeof *tmp);
    if (!tmp) { perror("tmp"); return -2; }

    size_t next_width = 4;
    cpu_merge_ladder(data, tmp, n, &next_width);

    if (next_width >= n) {
        free(tmp);
        return 0;
    }

    size_t start_k = next_width;

    int rc = bitonic_sort_opencl((int32_t*)data, n, start_k);

    free(tmp);
    return rc;
}


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

    // Init Rand
    srand((unsigned int)time(NULL));

    {
        // UINT SORT
        uint32_t *data = create_random_data_u32(N, UINT32_MAX);
        if(!data) {
            return -1;
        }
/**
        uint32_t *data_cpy = malloc(N * sizeof(uint32_t));
        if(!data_cpy) {
            return -1;
        }
        memcpy(data_cpy, data, N * sizeof(uint32_t));
        // Quicksort
        double q_start = getCurTime();
        qsort(data_cpy, N, sizeof(uint32_t), qsort_u32);
        double q_end = getCurTime();
        printf("Quick Sort in %0.8f -> %0.8f per elem!\n", q_end - q_start, ((q_end - q_start) / (double) N));
        free(data_cpy);
**/
        // Recursive Bitonic Sort
        double start = getCurTime();
        //bitonic_sort_u32(data, 0, N, true);
        //bitonic_sort_opencl(data, N, 0);
        //hybrid_tiered_sort_uint32(data, N);
        //simd_mergesort_uint32(data, N);
        //bitonic_sort_opencl(data, N, 0);
        //bitonic_cellsort(data, N);
        simd_mergesort_uint32(data, N);
        double end = getCurTime();
        if(!is_sorted_u32(data, N)) {
            printf("Array not sorted\n");
        } else {
            printf("Bitonic Sort in %0.8f -> %0.8f per elem!\n", end - start, ((end - start) / (double) N));
        }
        free(data);
    }
/**
    {
        // UINT SORT
        uint32_t *data = create_random_data_u32(N, UINT32_MAX);
        if(!data) {
            return -1;
        }
        uint32_t *data_cpy = malloc(N * sizeof(uint32_t));
        if(!data_cpy) {
            return -1;
        }
        memcpy(data_cpy, data, N * sizeof(uint32_t));
        // Quicksort
        double q_start = getCurTime();
        bitonic_sort_opencl(data, N, 0);
        double q_end = getCurTime();
        if(!is_sorted_u32(data, N)) {
            printf("Array not sorted\n");
        } else {
            printf("GPU MERGE Sort in %0.8f -> %0.8f per elem!\n", q_end - q_start, ((q_end - q_start) / (double) N));
        }
        free(data_cpy);

        free(data);
    }
    **/

    {
        // UINT SORT
        uint32_t *data = create_random_data_u32(N, UINT32_MAX);
        if(!data) {
            return -1;
        }

        // Recursive Bitonic Sort
        double start = getCurTime();
        bitonic_cellsort(data, N);
        double end = getCurTime();
        if(!is_sorted_u32(data, N)) {
            printf("Array not sorted\n");
        } else {
            printf("Hybrid Sort in %0.8f -> %0.8f per elem!\n", end - start, ((end - start) / (double) N));
        }
        free(data);
    }


    return 0;
}
