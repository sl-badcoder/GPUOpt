#ifndef BITONIC_GPU_H
#define BITONIC_GPU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void gpu_bitonic_sort_uint32(uint32_t *arr, int N);

#ifdef __cplusplus
}
#endif
#endif
