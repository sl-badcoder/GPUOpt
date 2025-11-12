#ifndef HYBRID_GPU_H
#define HYBRID_GPU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void hybrid_sort(uint32_t *arr, int N, int K_CPU_MAX);
//void hybrid_sort_chunk(uint32_t *arr, int N);

#ifdef __cplusplus
}
#endif
#endif
