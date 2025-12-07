//------------------------------------------------------------------------------------------------------------
#include <pthread.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
//------------------------------------------------------------------------------------------------------------
#ifndef MIN
#  define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif
//------------------------------------------------------------------------------------------------------------
#ifndef MAX
#  define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif
//------------------------------------------------------------------------------------------------------------
static uint32_t        *g_data;
static size_t           g_N;
static int              g_P;
static size_t           g_chunk;
static pthread_barrier_t g_barrier;
//------------------------------------------------------------------------------------------------------------
static inline void bitonic_merge_scalar(uint32_t *a, size_t n, bool asc) {
    if (n <= 1) return;
    size_t half = n >> 1;
    for (size_t i = 0; i < half; ++i) {
        bool cond = asc ? (a[i] > a[i+half]) : (a[i] < a[i+half]);
        if (cond) { uint32_t tmp=a[i]; a[i]=a[i+half]; a[i+half]=tmp; }
    }
    bitonic_merge_scalar(a,        half, asc);
    bitonic_merge_scalar(a+half,   half, asc);
}
//------------------------------------------------------------------------------------------------------------
static void bitonic_sort_scalar(uint32_t *a, size_t n, bool asc) {
    if (n <= 1) return;
    size_t half = n >> 1;
    bitonic_sort_scalar(a,        half, true );
    bitonic_sort_scalar(a+half,   half, false);
    bitonic_merge_scalar(a, n, asc);
}
//------------------------------------------------------------------------------------------------------------
static inline __m256i bitonic_sort8_u32(__m256i v) {
    __m256i sh1 = _mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(4,5,6,7,0,1,2,3));
    __m256i lo1 = _mm256_min_epu32(v, sh1);
    __m256i hi1 = _mm256_max_epu32(v, sh1);

    v = _mm256_blend_epi32(lo1, hi1, 0b11110000);
    __m256i sh2 = _mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(2,3,0,1,6,7,4,5));
    __m256i lo2 = _mm256_min_epu32(v, sh2);
    __m256i hi2 = _mm256_max_epu32(v, sh2);

    v = _mm256_blend_epi32(lo2, hi2, 0b11001100);
    __m256i sh3 = _mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(1,0,3,2,5,4,7,6));
    __m256i lo3 = _mm256_min_epu32(v, sh3);
    __m256i hi3 = _mm256_max_epu32(v, sh3);
    v = _mm256_blend_epi32(lo3, hi3, 0b10101010);
    return v;
}
//------------------------------------------------------------------------------------------------------------
static void chunk_sort_simd(uint32_t *ptr, size_t len, bool asc) {
    size_t i = 0;
    for (; i + 7 < len; i += 8) {
        __m256i v = _mm256_loadu_si256((__m256i*)(ptr + i));
        v = bitonic_sort8_u32(v);
        if (!asc) {
            v = _mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(7,6,5,4,3,2,1,0));
        }
        _mm256_storeu_si256((__m256i*)(ptr + i), v);
    }
    if (i < len) bitonic_sort_scalar(ptr + i, len - i, asc);

    for (size_t block = 8; block < len; block <<= 1) {
        for (size_t off = 0; off < len; off += 2*block) {
            size_t nmerge = (off + 2*block <= len) ? 2*block : len - off;
            bitonic_merge_scalar(ptr + off, nmerge, asc);
        }
    }
}
//------------------------------------------------------------------------------------------------------------
static void *cellsort_thread(void *arg) {
    int tid = (int)(intptr_t)arg;
    size_t start = (size_t)tid * g_chunk;
    bool asc_local = ((tid & 1) == 0);

    chunk_sort_simd(g_data + start, g_chunk, asc_local);
    pthread_barrier_wait(&g_barrier);

    for (size_t k = 2; k <= g_N; k <<= 1) {
        for (size_t j = k >> 1; j >= 1; j >>= 1) {
            pthread_barrier_wait(&g_barrier);
            if (j >= 8) {
                for (size_t idx = start; idx < start + g_chunk; idx += 8) {
                    size_t partner = idx ^ j;
                    if (partner < idx || partner >= g_N) continue;
                    bool asc = ((idx & k) == 0);
                    __m256i a = _mm256_loadu_si256((__m256i*)(g_data + idx));
                    __m256i b = _mm256_loadu_si256((__m256i*)(g_data + partner));
                    __m256i vmin = asc ? _mm256_min_epu32(a,b) : _mm256_max_epu32(a,b);
                    __m256i vmax = asc ? _mm256_max_epu32(a,b) : _mm256_min_epu32(a,b);
                    _mm256_storeu_si256((__m256i*)(g_data + idx),     vmin);
                    _mm256_storeu_si256((__m256i*)(g_data + partner), vmax);
                }
            } else {
                for (size_t idx = start; idx < start + g_chunk; ++idx) {
                    size_t partner = idx ^ j;
                    if (partner < idx || partner >= g_N) continue;
                    bool asc = ((idx & k) == 0);
                    uint32_t a = g_data[idx];
                    uint32_t b = g_data[partner];
                    if ((a > b) == asc) {
                        g_data[idx] = b;
                        g_data[partner] = a;
                    }
                }
            }
        }
    }
    return NULL;
}
//------------------------------------------------------------------------------------------------------------
void bitonic_cellsort(uint32_t *data, size_t N) {
    g_data = data;
    g_N    = N;
    int cores = sysconf(_SC_NPROCESSORS_ONLN);
    g_P = 1;
    while ((g_P << 1) <= cores && (N % (g_P << 1)) == 0) g_P <<= 1;
    g_chunk = N / g_P;

    pthread_barrier_init(&g_barrier, NULL, g_P);
    pthread_t th[g_P];
    for (int t = 0; t < g_P; ++t)
        pthread_create(&th[t], NULL, cellsort_thread, (void*)(intptr_t)t);
    for (int t = 0; t < g_P; ++t) pthread_join(th[t], NULL);
    pthread_barrier_destroy(&g_barrier);
}
//------------------------------------------------------------------------------------------------------------
static inline void cas_pair(uint32_t *a, uint32_t *b, bool asc)
{
    uint32_t x = *a, y = *b;
    if ( (x > y) == asc ) { *a = y; *b = x; }
}
//------------------------------------------------------------------------------------------------------------
static inline void cas_u32(uint32_t *a, uint32_t *b, bool asc)
{
    uint32_t x = *a, y = *b;
    if ( (x > y) == asc ) { *a = y; *b = x; }
}
//------------------------------------------------------------------------------------------------------------
static void *cellsort_thread_bitonic(void *arg)
{
    const int    tid    = (int)(intptr_t)arg;
    const size_t first  = (size_t)tid * g_chunk;
    const size_t last   = first + g_chunk;

    const bool asc_local = ((tid & 1) == 0);
    chunk_sort_simd(g_data + first, g_chunk, asc_local);
    pthread_barrier_wait(&g_barrier);

    for (size_t k = 2; k <= g_N; k <<= 1) {

        for (size_t j = k >> 1; j >= 1; j >>= 1) {

            pthread_barrier_wait(&g_barrier);

            for (size_t idx = first; idx < last; ++idx) {

                size_t partner = idx ^ j;
                if (partner < idx || partner >= g_N) continue;

                bool asc = ((idx & k) == 0);
                cas_u32(&g_data[idx], &g_data[partner], asc);
            }

            pthread_barrier_wait(&g_barrier);
        }
    }
    return NULL;
}
//------------------------------------------------------------------------------------------------------------
void bitonic_cellsort_oddeven(uint32_t *data, size_t N)
{
    g_data = data;  g_N = N;

    int hw = sysconf(_SC_NPROCESSORS_ONLN);
    g_P = 1;  while ((g_P<<1)<=hw && (N%(g_P<<1))==0) g_P <<= 1;
    g_chunk = N / g_P;

    pthread_barrier_init(&g_barrier, NULL, g_P);
    pthread_t th[g_P];
    for (int t=0;t<g_P;++t)
        pthread_create(&th[t],NULL,cellsort_thread_bitonic,(void*)(intptr_t)t);
    for (int t=0;t<g_P;++t) pthread_join(th[t],NULL);
    pthread_barrier_destroy(&g_barrier);
}
//------------------------------------------------------------------------------------------------------------