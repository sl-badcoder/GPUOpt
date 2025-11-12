#ifndef __BITONIC_H__
#define __BITONIC_H__

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

void bitonic_sort_u32(uint32_t *data, size_t start, size_t num, bool direction);
void bitonic_sort_float(float *data, size_t start, size_t num, bool direction);
#endif