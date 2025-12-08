//------------------------------------------------------------------------------------------------------------
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
//------------------------------------------------------------------------------------------------------------
#ifndef MIN
#  define MIN(a,b) ((a)<(b)?(a):(b))
#endif
//------------------------------------------------------------------------------------------------------------
static int cmp_uint32(const void *a, const void *b) {
    uint32_t aa = *(const uint32_t *) a, bb = *(const uint32_t *) b;
    return (aa > bb) - (aa < bb);
}
//------------------------------------------------------------------------------------------------------------
static inline __m128i sort4_epi32(__m128i v) {
    __m128i t = _mm_shuffle_epi32(v,_MM_SHUFFLE(2, 3, 0, 1));
    __m128i lo = _mm_min_epu32(v, t), hi = _mm_max_epu32(v, t);
    v = _mm_unpacklo_epi64(lo, hi);
    t = _mm_shuffle_epi32(v,_MM_SHUFFLE(1, 0, 3, 2));
    lo = _mm_min_epu32(v, t);
    hi = _mm_max_epu32(v, t);
    v = _mm_unpacklo_epi64(lo, hi);
    t = _mm_shuffle_epi32(v,_MM_SHUFFLE(2, 3, 0, 1));
    lo = _mm_min_epu32(v, t);
    hi = _mm_max_epu32(v, t);
    return _mm_unpacklo_epi64(lo, hi);
}
//------------------------------------------------------------------------------------------------------------
// SIMD 8-WIDE IMPLEMENTATION -> different to paper (only 4)
//------------------------------------------------------------------------------------------------------------
static inline __m256i sort8_epi32(__m256i v)
{
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);

    lo = sort4_epi32(lo);
    hi = sort4_epi32(hi);

    v  = _mm256_set_m128i(hi, lo);

    __m256i t  = _mm256_permute2f128_si256(v, v, 0x01);
    __m256i min = _mm256_min_epu32(v, t);
    __m256i max = _mm256_max_epu32(v, t);
    v  = _mm256_blend_epi32(min, max, 0b11110000);

    t  = _mm256_shuffle_epi32(v, _MM_SHUFFLE(2,3,0,1));
    min = _mm256_min_epu32(v, t);
    max = _mm256_max_epu32(v, t);
    v  = _mm256_blend_epi32(min, max, 0b11001100);

    t  = _mm256_shuffle_epi32(v, _MM_SHUFFLE(1,0,3,2));
    min = _mm256_min_epu32(v, t);
    max = _mm256_max_epu32(v, t);
    v  = _mm256_blend_epi32(min, max, 0b10101010);

    t  = _mm256_shuffle_epi32(v, _MM_SHUFFLE(2,3,0,1));
    min = _mm256_min_epu32(v, t);
    max = _mm256_max_epu32(v, t);
    return _mm256_blend_epi32(min, max, 0b10101010);
}
//------------------------------------------------------------------------------------------------------------
static inline __m512i sort16_epi32(__m512i v)
{
    const __m512i idx1 = _mm512_set_epi32(
        14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1);
    const __m512i idx2 = _mm512_set_epi32(
        13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2);
    const __m512i idx4 = _mm512_set_epi32(
        11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4);
    const __m512i idx8 = _mm512_set_epi32(
        7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8);

    const __mmask16 m2  = 0xCCCC;
    const __mmask16 m4  = 0xF0F0;
    const __mmask16 m8  = 0xFF00;
    const __mmask16 m1  = 0xAAAA;

    __m512i t, lo, hi;

    t  = _mm512_permutexvar_epi32(idx1, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v  = _mm512_mask_mov_epi32(lo, m1, hi);

    t  = _mm512_permutexvar_epi32(idx2, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v  = _mm512_mask_mov_epi32(lo, m2, hi);

    t  = _mm512_permutexvar_epi32(idx1, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v  = _mm512_mask_mov_epi32(lo, m2, hi);

    t  = _mm512_permutexvar_epi32(idx4, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v  = _mm512_mask_mov_epi32(lo, m4, hi);

    t  = _mm512_permutexvar_epi32(idx2, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v  = _mm512_mask_mov_epi32(lo, m4, hi);

    t  = _mm512_permutexvar_epi32(idx1, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v  = _mm512_mask_mov_epi32(lo, m4, hi);

    t  = _mm512_permutexvar_epi32(idx8, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v  = lo;

    t  = _mm512_permutexvar_epi32(idx4, v);
    lo = _mm512_min_epu32(v, t);
    v  = lo;

    t  = _mm512_permutexvar_epi32(idx2, v);
    lo = _mm512_min_epu32(v, t);
    v  = lo;

    t  = _mm512_permutexvar_epi32(idx1, v);
    lo = _mm512_min_epu32(v, t);
    v  = lo;

    return v;
}
//------------------------------------------------------------------------------------------------------------
// first phase sorting small 8-wide runs talked about in report
//------------------------------------------------------------------------------------------------------------
void vector_presort(uint32_t *d, size_t n)
{
    const size_t end = n & ~(size_t)15;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < end; i += 16) {
        __m512i v = _mm512_loadu_si512((const void*)(d + i));
        v = sort16_epi32(v);
        _mm512_storeu_si512((void*)(d + i), v);
    }
    for (size_t j = (end ? end + 1 : (n > 0 ? 1 : 0)); j < n; ++j) {
        uint32_t key = d[j];
        size_t k = j;
        while (k && d[k - 1] > key) {
            d[k] = d[k - 1];
            --k;
        }
        d[k] = key;
    }
}
//------------------------------------------------------------------------------------------------------------
static inline void merge_scalar(uint32_t *dst,
                                const uint32_t *L, size_t nl,
                                const uint32_t *R, size_t nr) {
    size_t i = 0, j = 0, k = 0;
    while (i < nl && j < nr) {
        uint32_t a = L[i], b = R[j];
        int take = (a > b);
        dst[k++] = take ? b : a;
        i += !take;
        j += take;
    }
    if (i < nl) memcpy(dst + k, L + i, (nl - i) * 4);
    if (j < nr) memcpy(dst + k, R + j, (nr - j) * 4);
}
//------------------------------------------------------------------------------------------------------------
void simd_merge_pass_uint32(const uint32_t *src,
                            uint32_t *dst,
                            size_t width,
                            size_t n) {
    size_t chunks = (n + (2 * width - 1)) / (2 * width);
//------------------------------------------------------------------------------------------------------------
#pragma omp parallel for schedule(static)
    for (ptrdiff_t c = 0; c < (ptrdiff_t) chunks; ++c) {
        size_t first = c * 2 * width;
        size_t mid = MIN(first+width, n);
        size_t last = MIN(first+2*width, n);
        //------------------------------------------------------------------------------------------------------------
        if (mid == last) {
            memcpy(dst + first, src + first, (last - first) * 4);
            continue;
        }
        merge_scalar(dst + first,
                     src + first, mid - first,
                     src + mid, last - mid);
    }
}
//------------------------------------------------------------------------------------------------------------
// Bottom-up Merge (second phase talked in report)
// space complexity: O(n) is way to expensive 
// out-of-place is not the best
//------------------------------------------------------------------------------------------------------------
static void bottom_up_mergesort(uint32_t *data, uint32_t *tmp, size_t n) {
    const size_t base =  8;
    uint32_t *src = data, *dst = tmp;
    //------------------------------------------------------------------------------------------------------------
    for (size_t w = base; w < n; w <<= 1) {
        simd_merge_pass_uint32(src, dst, w, n);
        uint32_t *s = src;
        src = dst;
        dst = s;
    }
    if (src != data) memcpy(data, src, n * 4);
}
//------------------------------------------------------------------------------------------------------------
static void bottom_up_mergesort_k(uint32_t *data, uint32_t *tmp, size_t n, size_t k) {
    const size_t base =  8;
    uint32_t *src = data, *dst = tmp;
    //------------------------------------------------------------------------------------------------------------
    for (size_t w = base; w < k; w <<= 1) {
        simd_merge_pass_uint32(src, dst, w, n);
        uint32_t *s = src;
        src = dst;
        dst = s;
    }
    if (src != data) memcpy(data, src, n * 4);
}
//------------------------------------------------------------------------------------------------------------
void simd_mergesort_uint32(uint32_t *data, size_t n) {
    if (n < 2) return;
    size_t bytes = n * (size_t) sizeof(uint32_t);
    vector_presort(data, n);
    //------------------------------------------------------------------------------------------------------------
    // Align data to cache size
    uint32_t *tmp;
    int err = posix_memalign((void **) &tmp, 64, bytes);
    if (err != 0) {
        fprintf(stderr, "posix_memalign failed for %zu bytes: %s\n", bytes, strerror(err));
        exit(1);
    }
    //------------------------------------------------------------------------------------------------------------
    bottom_up_mergesort(data, tmp, n);
    free(tmp);
}
//------------------------------------------------------------------------------------------------------------
void simd_mergesort_uint32_k(uint32_t *data, size_t n, size_t k) {
    if (n < 2) return;
    vector_presort(data, n);
    if(k<=16){
        //printf("k: %d", k);
        return;
    }
    //------------------------------------------------------------------------------------------------------------
    // Align data to cache size
    uint32_t *tmp;
    posix_memalign((void **) &tmp, 64, n * 4);
    //------------------------------------------------------------------------------------------------------------
    bottom_up_mergesort_k(data, tmp, n, k);
    free(tmp);
}
//------------------------------------------------------------------------------------------------------------
static void bitonic_step_simd512(uint32_t *data, size_t n,
                                       size_t k)
{
    const size_t VEC = 16;
    size_t j = k >> 1;

    #pragma omp for schedule(static)
    for (ptrdiff_t block = 0; block < (ptrdiff_t)n; block += (ptrdiff_t)k) {
        size_t b = (size_t)block;

        if (b >= n) continue;
        size_t span = n - b;

        if (span <= j) continue;

        size_t max_pairs = span - j;
        size_t eff_j = (max_pairs < j) ? max_pairs : j;
        if (eff_j == 0) continue;

        int ascending = ((b & k) == 0);

        size_t l = 0;

        for (; l + VEC <= eff_j; l += VEC) {
            size_t i  = b + l;
            size_t ix = i + j;

            __m512i va = _mm512_loadu_si512((const void*)(data + i));
            __m512i vb = _mm512_loadu_si512((const void*)(data + ix));

            __m512i vmin = _mm512_min_epu32(va, vb);
            __m512i vmax = _mm512_max_epu32(va, vb);

            if (ascending) {
                _mm512_storeu_si512((void*)(data + i),      vmin);
                _mm512_storeu_si512((void*)(data + ix),     vmax);
            } else {
                _mm512_storeu_si512((void*)(data + i),      vmax);
                _mm512_storeu_si512((void*)(data + ix),     vmin);
            }
        }

        for (; l < eff_j; ++l) {
            size_t i  = b + l;
            size_t ix = i + j;
            if (ix >= n) break;

            uint32_t a = data[i];
            uint32_t bval = data[ix];

            if (ascending ? (a > bval) : (a < bval)) {
                data[i]   = bval;
                data[ix]  = a;
            }
        }
    }
}

static void bitonic_step_scalar(uint32_t *data, size_t n,
                                size_t j, size_t k)
{
    #pragma omp for schedule(static)
    for (ptrdiff_t ii = 0; ii < (ptrdiff_t)n; ++ii) {
        size_t i   = (size_t)ii;
        size_t ixj = i ^ j;
        if (ixj > i && ixj < n) {
            uint32_t a = data[i];
            uint32_t b = data[ixj];
            int ascending = ((i & k) == 0);
            if ((a > b) == ascending) {
                data[i]   = b;
                data[ixj] = a;
            }
        }
    }
}
//------------------------------------------------------------------------------------------------------------
void simd_bitonic_sort_uint32(uint32_t *data, size_t n)
{
    if (n < 2) return;
    //------------------------------------------------------------------------------------------------------------
    vector_presort(data, n);
    //------------------------------------------------------------------------------------------------------------
    #pragma omp parallel
    {
        for (size_t k = 2; k <= n; k <<= 1) {
            bitonic_step_simd512(data, n, k);
            for (size_t j = k >> 2; j > 0; j >>= 1) {
                bitonic_step_scalar(data, n, j, k);
            }
        }
    }
}

//------------------------------------------------------------------------------------------------------------
void simd_bitonic_sort_uint32_k(uint32_t *data, size_t n, size_t k_start)
{
    if (n < 2) return;
    if (k_start < 2) {
        simd_bitonic_sort_uint32(data, n);
        return;
    }
    #pragma omp parallel
    {
        for (size_t k = 2 * k_start; k <= n; k <<= 1) {
            bitonic_step_simd512(data, n, k);
            for (size_t j = k >> 2; j > 0; j >>= 1) {
                bitonic_step_scalar(data, n, j, k);
            }
        }
    }
}
//------------------------------------------------------------------------------------------------------------