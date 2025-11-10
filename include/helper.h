#ifndef __HELPER_H__
#define __HELPER_H__

#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <stdbool.h>

double getCurTime(void);
uint32_t* create_random_data_u32(size_t N, size_t MAX_VAL);
uint32_t* create_random_data_u32_pinned(size_t N, size_t MAX_VAL);
float* create_random_data_float(size_t N);
bool is_sorted_u32(uint32_t *data, size_t len);
bool is_sorted_float(float *data, size_t len);
int qsort_u32(const void *elem1, const void *elem2);
int qsort_float(const void *elem1, const void *elem2);

#endif