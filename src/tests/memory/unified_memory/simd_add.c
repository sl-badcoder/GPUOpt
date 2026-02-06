#include <immintrin.h>
#include <stddef.h>

void add1_simd(char *data, size_t N)
{
#if defined(__AVX2__)
    size_t i = 0;
    const __m256i ones = _mm256_set1_epi8(1); 

    for (; i + 31 < N; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(data + i));
        v = _mm256_add_epi8(v, ones);
        _mm256_storeu_si256((__m256i *)(data + i), v);
    }

    for (; i < N; ++i) {
        data[i] += 1;
    }
#else
    for (size_t i = 0; i < N; ++i) {
        data[i] += 1;
    }
#endif
}
