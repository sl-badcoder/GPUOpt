#ifndef BITONIC_GPU_H
#define BITONIC_GPU_H
#include <stddef.h>
#include <stdint.h>

int bitonic_sort_opencl(uint32_t *data, size_t N, size_t start_k);
int bitonic_sort_opencl_vec(uint32_t *data, size_t N);
int bitonic_gpu_optimized(uint32_t *a, size_t N);
int bitonic_sort_opencl_(uint32_t *a, size_t N);
int bitonic_sort_opencl_opt(int *arr, size_t size, size_t start_k);
int bitonic_sort_opencl_sub(int *arr, size_t size, size_t start_k);
#endif //BITONIC_GPU_H
