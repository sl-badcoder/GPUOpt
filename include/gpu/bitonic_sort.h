#ifndef BITONIC_SORT_H
#define BITONIC_SORT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void gpu_bitonic_sort_uint32(uint32_t *arr, int N);
void gpu_bitonic_sort_uint32_k(uint32_t *arr, int N, int k_start, bool MAPPED);
void gpu_bitonic_sort_uint32_chunk(uint32_t *arr, int N, cudaStream_t s);

#ifdef __cplusplus
}
#endif
#endif
