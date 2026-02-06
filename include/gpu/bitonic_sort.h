#ifndef BITONIC_SORT_H
#define BITONIC_SORT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void gpu_bitonic_sort_uint32(uint32_t *arr, size_t N);
void gpu_bitonic_sort_uint32_k(uint32_t *arr, size_t N, int k_start, bool MAPPED);
void gpu_bitonic_sort_uint32_chunk(uint32_t *arr, size_t N, cudaStream_t s);
void gpu_bitonic_sort_uint32_k_un(uint32_t *arr, size_t N, size_t k_start, cudaStream_t s);
//void gpu_bitonic_sort_uint32_k_un_b(uint32_t *arr, size_t N, size_t k_start, bool asc);

#ifdef __cplusplus
}
#endif
#endif
