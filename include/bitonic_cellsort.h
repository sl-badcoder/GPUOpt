#ifndef BITONIC_CELLSORT_H
#define BITONIC_CELLSORT_H
#include <stddef.h>
#include <stdint.h>

void bitonic_cellsort(uint32_t *data, size_t N);
void bitonic_cellsort_adap(uint32_t *data, size_t N);
void bitonic_cellsort_oddeven(uint32_t *data, size_t N);
#endif //BITONIC_CELLSORT_H
