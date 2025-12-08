#ifndef BITONIC_SIMD_MERGE_H
#define BITONIC_SIMD_MERGE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif


void simd_mergesort_uint32(uint32_t *data, size_t N);
void bitonic_sort(uint32_t *data, size_t n);
void vector_presort(uint32_t *data, size_t n);
void simd_merge_pass_uint32(const uint32_t *src,
                            uint32_t       *dst,
                            size_t          width,
                            size_t          n);
void simd_mergesort_uint32_k(uint32_t *data, size_t n, size_t k);
void simd_bitonic_sort_uint32(uint32_t *data, size_t n);
void simd_bitonic_sort_uint32_k(uint32_t *data, size_t n, size_t k_start);
#ifdef __cplusplus
}
#endif
#endif //BITONIC_SIMD_MERGE_H
