#include <stdio.h>
#include "helper.h"

// Returns the current time using CLOCK_MONOTONIC
double getCurTime(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec + t.tv_nsec*1e-9;
}

uint32_t* create_random_data_u32(size_t N, size_t MAX_VAL) {
    uint32_t* data = malloc(N * sizeof(uint32_t));
    if (!data) {
        perror("malloc failed");
        return NULL;
    }
    for(size_t i = 0; i < N; ++i) {
        data[i] = rand() % MAX_VAL;
    }
    return data;
}

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

bool is_sorted_u32(uint32_t *data, size_t len) {
    for(size_t i = 0; i < len - 1; ++i) {
        if(data[i] > data[i+1]) {
            printf("%lu: %u >  %lu: %u\n", i, data[i], i+1, data[i+1]);
            return false;
        }
    }
    return true;
}

bool is_sorted_float(float *data, size_t len) {
    for(size_t i = 0; i < len - 1; ++i) {
        if(data[i] > data[i+1]) {
            printf("%lu: %0.8f > %lu: %0.8f\n", i, data[i], i+1, data[i+1]);
            return false;
        }
    }
    return true;
}

int qsort_u32(const void *elem1, const void *elem2) {
    if(*((uint32_t*)elem1) > *((uint32_t*)elem2)) {
        return 1;
    } else if(*((uint32_t*)elem1) < *((uint32_t*)elem2)) {
        return -1;
    }
    return 0;
}

int qsort_float(const void *elem1, const void *elem2) {
    if(*((float*)elem1) > *((float*)elem2)) {
        return 1;
    } else if(*((float*)elem1) < *((float*)elem2)) {
        return -1;
    }
    return 0;
}