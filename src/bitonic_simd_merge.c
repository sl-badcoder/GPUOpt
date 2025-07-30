#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#ifndef MIN
#  define MIN(a,b) ((a)<(b)?(a):(b))
#endif

static int cmp_uint32(const void *a, const void *b) {
    uint32_t aa = *(const uint32_t *) a, bb = *(const uint32_t *) b;
    return (aa > bb) - (aa < bb);
}

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

// SIMD 8-WIDE IMPLEMENTATION -> different to paper (only 4)
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

// first phase sorting small 8-wide runs talked about in report
void vector_presort(uint32_t *d, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(d + i));
        v = sort8_epi32(v);
        _mm256_storeu_si256((__m256i *)(d + i), v);
    }
    for (size_t j = i + 1; j < n; ++j) {
        uint32_t key = d[j];
        size_t k = j;
        while (k && d[k - 1] > key) {
            d[k] = d[k - 1];
            --k;
        }
        d[k] = key;
    }
}

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

void simd_merge_pass_uint32(const uint32_t *src,
                            uint32_t *dst,
                            size_t width,
                            size_t n) {
    size_t chunks = (n + (2 * width - 1)) / (2 * width);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t c = 0; c < (ptrdiff_t) chunks; ++c) {
        size_t first = c * 2 * width;
        size_t mid = MIN(first+width, n);
        size_t last = MIN(first+2*width, n);

        if (mid == last) {
            memcpy(dst + first, src + first, (last - first) * 4);
            continue;
        }
        merge_scalar(dst + first,
                     src + first, mid - first,
                     src + mid, last - mid);
    }
}

// Bottom-up Merge (second phase talked in report)
static void bottom_up_mergesort(uint32_t *data, uint32_t *tmp, size_t n) {
    const size_t base =  8;
    uint32_t *src = data, *dst = tmp;

    for (size_t w = base; w < n; w <<= 1) {
        simd_merge_pass_uint32(src, dst, w, n);
        uint32_t *s = src;
        src = dst;
        dst = s;
    }
    if (src != data) memcpy(data, src, n * 4);
}

void simd_mergesort_uint32(uint32_t *data, size_t n) {
    if (n < 2) return;
    vector_presort(data, n);

    // Align data to cache size
    uint32_t *tmp;
    posix_memalign((void **) &tmp, 64, n * 4);

    bottom_up_mergesort(data, tmp, n);
    free(tmp);
}
