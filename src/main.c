#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

#include "bitonic.h"
#include "helper.h"

int main(int argc, char **argv) {
    size_t N;
    if(argc < 2) {
        printf("./sort [NUM_OF_ELEM]\n");
        return 0;
    }
    N = atol(argv[1]);
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

        // Recursive Bitonic Sort
        double start = getCurTime();
        bitonic_sort_u32(data, 0, N, true);
        double end = getCurTime();
        if(!is_sorted_u32(data, N)) {
            printf("Array not sorted\n");
        } else {
            printf("Bitonic Sort in %0.8f -> %0.8f per elem!\n", end - start, ((end - start) / (double) N));
        }
        free(data);
    }

    {
        // FLOAT SORT
        float *data = create_random_data_float(N);
        if(!data) {
            return -1;
        }
        float *data_cpy = malloc(N * sizeof(float));
        if(!data_cpy) {
            return -1;
        }
        memcpy(data_cpy, data, N * sizeof(float));
        // Quicksort
        double q_start = getCurTime();
        qsort(data_cpy, N, sizeof(uint32_t), qsort_float);
        double q_end = getCurTime();
        printf("Quick Sort in %0.8f -> %0.8f per elem!\n", q_end - q_start, ((q_end - q_start) / (double) N));
        free(data_cpy);

        // Recursive Bitonic Sort
        double start = getCurTime();
        bitonic_sort_float(data, 0, N, true);
        double end = getCurTime();
        if(!is_sorted_float(data, N)) {
            printf("Array not sorted\n");
        } else {
            printf("Bitonic Sort in %0.8f -> %0.8f per elem!\n", end - start, ((end - start) / (double) N));
        }
        free(data);
    }

    return 0;
}