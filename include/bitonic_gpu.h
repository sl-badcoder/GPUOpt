#ifndef BITONIC_GPU_H
#define BITONIC_GPU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void gpu_bitonic_sort_uint32(uint32_t *arr, int N);
void gpu_bitonic_sort_uint32_k(uint32_t *arr, int N, int k_start);
void hybrid_sort(uint32_t *arr, int N, int K_CPU_MAX);

#ifdef __cplusplus
}
#endif
#endif
