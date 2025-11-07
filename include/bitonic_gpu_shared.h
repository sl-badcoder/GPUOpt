#ifndef BITONIC_GPU_SHARED_H
#define BITONIC_GPU_SHARED_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void gpu_bitonic_sort_uint32_shared(uint32_t *arr, int N);

#ifdef __cplusplus
}
#endif
#endif
