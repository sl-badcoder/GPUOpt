//------------------------------------------------------------------------------------------------------------
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <stdint.h>
//------------------------------------------------------------------------------------------------------------
#include "core/helper.h"
//------------------------------------------------------------------------------------------------------------
// Returns the current time using CLOCK_MONOTONIC
double getCurTime(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec + t.tv_nsec*1e-9;
}
//------------------------------------------------------------------------------------------------------------
uint32_t* create_random_data_u32(size_t N, size_t MAX_VAL) {
    uint32_t* data;
    posix_memalign((void **) &data, 64, N * sizeof(uint32_t));
    if (!data) {
        perror("malloc failed");
        return NULL;
    }
    for(size_t i = 0; i < N; ++i) {
        data[i] = rand() % MAX_VAL;
    }
    return data;
}
//------------------------------------------------------------------------------------------------------------
// create data for pinned memory tests
//------------------------------------------------------------------------------------------------------------
uint32_t* create_random_data_u32_pinned(long long N, size_t MAX_VAL) {
    uint32_t *data = NULL;
    long long bytes = N *(long long)  sizeof(uint32_t);
    size_t tmp = N * (long long) sizeof(uint32_t) / (long long)(1024.0 * 1024.0);
    printf("[ALLOCATE] ARRAY OF SIZE: %lld MiB\n", tmp);
    CHECK_CUDA(cudaMallocHost((void**)&data, bytes)); 
    
    for (size_t i = 0; i < N; ++i) {
        data[i] = (uint32_t)(rand() % MAX_VAL);
    }
    return data; 
}
//------------------------------------------------------------------------------------------------------------
// create data for mapped memory tests
//------------------------------------------------------------------------------------------------------------
uint32_t* create_random_data_u32_mapped(long long N, size_t MAX_VAL) {
    uint32_t *data = NULL;
    long long bytes = N * (long long) sizeof(uint32_t);
    long long tmp = N * (long long) sizeof(uint32_t) / (long long)(1024.0 * 1024.0);
    printf("[ALLOCATE] ARRAY OF SIZE: %lld MiB\n", tmp);
    CHECK_CUDA(cudaHostAlloc((void**)&data, bytes, cudaHostAllocMapped)); 
    
    for (size_t i = 0; i < N; ++i) {
        data[i] = (uint32_t)(rand() % MAX_VAL);
    }
    return data; 
}
//------------------------------------------------------------------------------------------------------------
uint32_t* create_random_data_u32_unified(long long N, size_t MAX_VAL) {
    uint32_t *data = NULL;
    long long bytes = N *(long long)  sizeof(uint32_t);
   
    long long tmp = N * (long long) sizeof(uint32_t) / (long long)(1024.0 * 1024.0);
   
    printf("[ALLOCATE] ARRAY OF SIZE: %lld MiB\n", tmp);
   
    CHECK_CUDA(cudaMallocManaged((void**)&data, bytes, cudaMemAttachGlobal)); 

    for (size_t i = 0; i < N; ++i) {
        data[i] = (uint32_t)(rand() % MAX_VAL);
    }
    return data; 
}
//------------------------------------------------------------------------------------------------------------
float* create_random_data_float(size_t N) {
    float* data = malloc(N * sizeof(float));
    if (!data) {
        perror("malloc failed");
        return NULL;
    }
    for(size_t i = 0; i < N; ++i) {
        data[i] = (float)rand() / (float)RAND_MAX;
    }
    return data;
}
//------------------------------------------------------------------------------------------------------------
bool is_sorted_u32(uint32_t *data, size_t len) {
    for(size_t i = 0; i < len - 1; ++i) {
        if(data[i] > data[i+1]) {
            printf("%lu: %u >  %lu: %u\n", i, data[i], i+1, data[i+1]);
            return false;
        }
    }
    return true;
}
//------------------------------------------------------------------------------------------------------------
bool is_sorted_float(float *data, size_t len) {
    for(size_t i = 0; i < len - 1; ++i) {
        if(data[i] > data[i+1]) {
            printf("%lu: %0.8f > %lu: %0.8f\n", i, data[i], i+1, data[i+1]);
            return false;
        }
    }
    return true;
}
//------------------------------------------------------------------------------------------------------------
int qsort_u32(const void *elem1, const void *elem2) {
    if(*((uint32_t*)elem1) > *((uint32_t*)elem2)) {
        return 1;
    } else if(*((uint32_t*)elem1) < *((uint32_t*)elem2)) {
        return -1;
    }
    return 0;
}
//------------------------------------------------------------------------------------------------------------
int qsort_float(const void *elem1, const void *elem2) {
    if(*((float*)elem1) > *((float*)elem2)) {
        return 1;
    } else if(*((float*)elem1) < *((float*)elem2)) {
        return -1;
    }
    return 0;
}
//------------------------------------------------------------------------------------------------------------
void reverse_block_u32_avx512(uint32_t *a, int start, int K) {
    if (K <= 1) return;

    const __m512i idx_rev = _mm512_set_epi32(
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    );

    int i = 0;
    int j = K;

    while (j - i >= 16) {
        __m512i L = _mm512_loadu_si512((const void*)(a + start + i));
        __m512i R = _mm512_loadu_si512((const void*)(a + start + (j - 16)));

        const __m512i idx = _mm512_set_epi32(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15
        );
        const __m512i idx_r = _mm512_set_epi32(
            15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0
        );

        __m512i L_rev = _mm512_permutexvar_epi32(idx_r, L);
        __m512i R_rev = _mm512_permutexvar_epi32(idx_r, R);

        _mm512_storeu_si512((void*)(a + start + i),        R_rev);
        _mm512_storeu_si512((void*)(a + start + (j - 16)), L_rev);

        i += 16;
        j -= 16;
    }

    int left = start + i;
    int right = start + j - 1;
    while (left < right) {
        uint32_t t = a[left]; a[left] = a[right]; a[right] = t;
        ++left; --right;
    }
}
//------------------------------------------------------------------------------------------------------------
void make_alternating_runs(uint32_t *a, int N, int K) {
    if (K <= 1 || N <= K) return;

    int count = (N - K) / (2 * K);
    if (count <= 0) return;

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < count; ++m) {
        int base = K + (2 * K) * m;
        reverse_block_u32_avx512(a, base, K);
    }
}
//------------------------------------------------------------------------------------------------------------
#define CHECK_CUDA(call) do {                                         \
  cudaError_t _e = (call);                                            \
  if (_e != cudaSuccess) {                                            \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
            cudaGetErrorString(_e));                                  \
    exit(1);                                                          \
  }                                                                   \
} while (0)
//------------------------------------------------------------------------------------------------------------
size_t L3CACHE = 33554432;
//------------------------------------------------------------------------------------------------------------
void warmup_cache(){
    char val[33554432] = {0};
    for(int i=0;i<L3CACHE;i++){
        val[i] += 1;
    }
}
//------------------------------------------------------------------------------------------------------------