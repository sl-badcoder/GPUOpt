#include <unistd.h>

#include "cpu/bitonic.h"

#define SWAP(data, idx_i, idx_j) \
    do { \
        typeof((data)[0]) _tmp = (data)[idx_i]; \
        (data)[idx_i] = (data)[idx_j]; \
        (data)[idx_j] = _tmp; \
    } while (0)

#define COMP_AND_SWAP(data, idx_i, idx_j, direction) \
    do { \
        if(((data)[idx_i] > (data)[idx_j]) == (direction)) { \
            SWAP(data, idx_i, idx_j); \
        } \
    } while (0) 

static void bitonic_merge_u32(uint32_t *data, size_t start, size_t num, bool direction) {
    if(num > 1) {
        size_t buck = num / 2;
        for(size_t i = start; i < start + buck; i++) {
            COMP_AND_SWAP(data, i, i + buck, direction);
        }
        bitonic_merge_u32(data, start, buck, direction);
        bitonic_merge_u32(data, start + buck, buck, direction);
    }
}

static void bitonic_merge_float(float *data, size_t start, size_t num, bool direction) {
    if(num > 1) {
        size_t buck = num / 2;
        for(size_t i = start; i < start + buck; i++) {
            COMP_AND_SWAP(data, i, i + buck, direction);
        }
        bitonic_merge_float(data, start, buck, direction);
        bitonic_merge_float(data, start + buck, buck, direction);
    }
}

/*
 * Creeate Bitonic Sequence and sort it
 * direction: true -> ascending, false -> descending
 * WARNING: Recursive
*/
void bitonic_sort_u32(uint32_t *data, size_t start, size_t num, bool direction) {
    if(num > 1) {
        // Split input in two parts
        size_t buck = num / 2;
        bitonic_sort_u32(data, start, buck, true);
        bitonic_sort_u32(data, start + buck, buck, false);
        bitonic_merge_u32(data, start, num, direction);
    }
}

/*
 * Creeate Bitonic Sequence and sort it
 * direction: true -> ascending, false -> descending
 * WARNING: Recursive
*/
void bitonic_sort_float(float *data, size_t start, size_t num, bool direction) {
    if(num > 1) {
        // Split input in two parts
        size_t buck = num / 2;
        bitonic_sort_float(data, start, buck, true);
        bitonic_sort_float(data, start + buck, buck, false);
        bitonic_merge_float(data, start, num, direction);
    }
}