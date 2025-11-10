//------------------------------------------------------------------------------------------------------------
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
//------------------------------------------------------------------------------------------------------------
#include "bitonic_gpu.h"
#include "bitonic_simd_merge.h"
//------------------------------------------------------------------------------------------------------------
#define _POSIX_C_SOURCE 199309L
//------------------------------------------------------------------------------------------------------------
// This is a baseline GPU implementation of bitonic sort which should be later used in the context
// of a hybrid execution model where we switch between a CPU baseline implementation 
// and a GPU baseline implementation such that we explore the best of both worlds. 
// Optimizations:
// - Shared memory should be explored properly
// - Reduce number of page misses - chunk permutation
// - work before writing
//------------------------------------------------------------------------------------------------------------
// First sort blocks which fit in shared memory 
template<int TILE>
__global__ void bitonic_block_sort(uint32_t *d, int n) {
    extern __shared__ uint32_t s[]; // define array in shared memory 
    int base = blockIdx.x * TILE;
    int tid  = threadIdx.x;
    //------------------------------------------------------------------------------------------------------------
    auto smem_idx = [&](int i){ return i + (i >> 5); };
    //------------------------------------------------------------------------------------------------------------
    for (int i = tid; i < TILE; i += blockDim.x) {
        int gi = base + i;
        s[smem_idx(i)] = (gi < n) ? d[gi] : 0xFFFFFFFFu;
    }
    __syncthreads();
    //------------------------------------------------------------------------------------------------------------
    for (int k = 2; k <= TILE; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < TILE; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    uint32_t a = s[smem_idx(i)];
                    uint32_t b = s[smem_idx(ixj)];
                    bool up = ((i & k) == 0);
                    if ((a > b) == up) {
                        s[smem_idx(i)]   = b;
                        s[smem_idx(ixj)] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    //------------------------------------------------------------------------------------------------------------
    bool desc = ((blockIdx.x & 1) != 0);
    if(desc){
        for(int i = tid; i < TILE/2; i+=blockDim.x){
            int a = i;
            int b = TILE - 1 - i;
            uint32_t va = s[smem_idx(a)];
            uint32_t vb = s[smem_idx(b)];
            s[smem_idx(a)] = vb;
            s[smem_idx(b)] = va;
        }
        __syncthreads();
    }
    //------------------------------------------------------------------------------------------------------------
    for (int i = tid; i < TILE; i += blockDim.x) {
        int gi = base + i;
        if (gi < n) d[gi] = s[smem_idx(i)];
    }
    //------------------------------------------------------------------------------------------------------------
}
//------------------------------------------------------------------------------------------------------------
template<int TILE>
__global__ void bitonic_shared(uint32_t* data, int k, int n){
    extern __shared__ uint32_t s[]; // shared memory array
    int max_w = 2 * TILE;
    int base = blockIdx.x * (2 * TILE);
    int tid = threadIdx.x;

    // propely access the shared memory array on the GPU
    auto smem_idx = [&](int i){
        return i + (i>>5);
    };

    // load data from data array
    for(int i = tid; i < max_w; i+=blockDim.x){
        int gi = base + i;
        s[smem_idx(i)] = (gi < n) ? data[gi] : 0xFFFFFFFFu;
    }
    __syncthreads();
    //------------------------------------------------------------------------------------------------------------
    // normal bitonic sort logic updated to global 
    //------------------------------------------------------------------------------------------------------------
    for(int j = min(TILE, k >> 1); j >0 ; j >>=1){
        for(int i = tid; i < max_w; i+=blockDim.x){
            int gi = base + i;
            int g_ixj = gi ^ j;  
            if (g_ixj >= base && g_ixj < (base + max_w) && g_ixj > gi) {
                int l_i = i;
                int l_ixj = g_ixj - base;
                uint32_t a = s[smem_idx(l_i)];
                uint32_t b = s[smem_idx(l_ixj)];
                bool ascending = ((gi & k) == 0);
                if ((a > b) == ascending) {
                    s[smem_idx(l_i)]   = b;
                    s[smem_idx(l_ixj)] = a;
                }
            }
        }
            __syncthreads();
    }
    //------------------------------------------------------------------------------------------------------------
    // store back window
    //------------------------------------------------------------------------------------------------------------
    for(int i = tid; i < max_w; i+=blockDim.x){
        int gi = base + i;
        if(gi<n) data[gi] = s[smem_idx(i)];
    }
}
//------------------------------------------------------------------------------------------------------------
__global__ void bitonic_step(uint32_t *data, int j, int k, int n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned)n) return;

    unsigned ixj = i ^ (unsigned)j;  
    if (ixj > i && ixj < (unsigned)n) {
        uint32_t a = data[i];
        uint32_t b = data[ixj];
        bool ascending = ((i & (unsigned)k) == 0u);
        if ((a > b) == ascending) {
            data[i]   = b;
            data[ixj] = a;
        }
    }
}
// like bitonic step but with for loop. 
__global__ void bitonic_step_for(uint32_t *data, int j, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(; i <n; i+= stride){
        int ixj = i ^ j;  
        if (ixj > i && ixj < n) {
            uint32_t a = data[i];
            uint32_t b = data[ixj];
            bool ascending = ((i & k) == 0u);
            if ((a > b) == ascending) {
                data[i]   = b;
                data[ixj] = a;
            }
        }
    }

}
//------------------------------------------------------------------------------------------------------------
// IDEAS: 
// get size of shared memory 
// size of array / shared memory
// split up array such that it fits in shaerd.
// when sorted in shared and shared memory is used fully switch back to global. 
//------------------------------------------------------------------------------------------------------------
extern "C" void gpu_bitonic_sort_uint32(uint32_t *arr, int N) {
    if (N <= 1) return;
    //------------------------------------------------------------------------------------------------------------
    // allocate the buffer array
    //------------------------------------------------------------------------------------------------------------
    uint32_t *dbuf = NULL;
    (cudaMalloc((void**)&dbuf, (size_t)N * sizeof(uint32_t)));
    (cudaMemcpy(dbuf, arr, (size_t)N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    //------------------------------------------------------------------------------------------------------------
    /**cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);**/
    //------------------------------------------------------------------------------------------------------------
    // define block size
    const int BLOCK = 1024;
    const int TILE = 4 * BLOCK;
    dim3 block(BLOCK);
    size_t shmem_block = (TILE + TILE/32) * sizeof(uint32_t);
    //------------------------------------------------------------------------------------------------------------
    //cudaEventRecord(start);
    int grid_b = (N + TILE - 1)/ TILE;
    bitonic_block_sort<TILE><<<grid_b, BLOCK, shmem_block>>>(dbuf, N);
    /**cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU Time: %.3f ms\n", ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);**/
    //------------------------------------------------------------------------------------------------------------
    // need to think on how to properly make use of 
    // right now just simply launch one bitonic step is really naive. 
    // maybe split up multiple steps such that they can exploit shared memory
    //------------------------------------------------------------------------------------------------------------ 
    /**cudaEvent_t start_shared, stop_shared;
    cudaEventCreate(&start_shared);
    cudaEventCreate(&stop_shared);
    cudaEventRecord(start_shared);**/
    // need to rethink this for loop -> launches way to much kernels
    for (int k = 2 * TILE; k <= N; k <<= 1) {
        //------------------------------------------------------------------------------------------------------------ 
        // if it fits in shared memory do the work in shared memory
        //------------------------------------------------------------------------------------------------------------ 
        for (int j = k >> 1; j > TILE; j >>= 1) {
        //------------------------------------------------------------------------------------------------------------ 
        // fallback option if it does not fit in shared memory anymore
        //------------------------------------------------------------------------------------------------------------ 
            {
                int grid = (N + BLOCK - 1) / BLOCK;
                //printf("grid: %d ", grid);
                bitonic_step_for<<<grid, BLOCK>>>(dbuf, j, k, N);
                //(cudaGetLastError());
            }
        }
        {
            //int max_w = 2 * TILE;
            int grid = (N + (2*TILE) - 1) / (2*TILE);
            bitonic_shared<TILE><<<grid, BLOCK, ((2* TILE)+(2*TILE)/32)*sizeof(uint32_t)>>>(dbuf,k, N);
        }

    }
    //------------------------------------------------------------------------------------------------------------ 
    /**cudaDeviceSynchronize();   
    cudaEventRecord(stop_shared);
    cudaEventSynchronize(stop_shared);
    //------------------------------------------------------------------------------------------------------------ 
    cudaEventElapsedTime(&ms, start_shared, stop_shared);
    printf("GPU Time: %.3f ms\n", ms);
    cudaEventDestroy(start_shared);
    cudaEventDestroy(stop_shared);**/
    //------------------------------------------------------------------------------------------------------------
    (cudaMemcpy(arr, dbuf, (size_t)N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    (cudaFree(dbuf));
    //------------------------------------------------------------------------------------------------------------
}


extern "C" void gpu_bitonic_sort_uint32_k(uint32_t *arr, int N, int k_start) {
    if (N <= 1) return;
    //------------------------------------------------------------------------------------------------------------
    // allocate the buffer array
    //------------------------------------------------------------------------------------------------------------
    uint32_t *dbuf = NULL;
    cudaHostRegister(arr, N * (sizeof(uint32_t)), cudaHostRegisterMapped);
    cudaHostGetDevicePointer(&dbuf, arr, 0);
    //(cudaMalloc((void**)&dbuf, (size_t)N * sizeof(uint32_t)));
    //(cudaMemcpy(dbuf, arr, (size_t)N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    //------------------------------------------------------------------------------------------------------------
    /**cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);**/
    //------------------------------------------------------------------------------------------------------------
    // define block size
    const int BLOCK = 1024;
    const int TILE = 4 * BLOCK;
    dim3 block(BLOCK);
    size_t shmem_block = (TILE + TILE/32) * sizeof(uint32_t);
    //------------------------------------------------------------------------------------------------------------
    //cudaEventRecord(start);
    int grid_b = (N + TILE - 1)/ TILE;
    //bitonic_block_sort<TILE><<<grid_b, BLOCK, shmem_block>>>(dbuf, N);
    /**cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU Time: %.3f ms\n", ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);**/
    //------------------------------------------------------------------------------------------------------------
    // need to think on how to properly make use of 
    // right now just simply launch one bitonic step is really naive. 
    // maybe split up multiple steps such that they can exploit shared memory
    //------------------------------------------------------------------------------------------------------------ 
    /**cudaEvent_t start_shared, stop_shared;
    cudaEventCreate(&start_shared);
    cudaEventCreate(&stop_shared);
    cudaEventRecord(start_shared);**/
    // need to rethink this for loop -> launches way to much kernels
    for (int k = 2 * k_start; k <= N; k <<= 1) {
        //------------------------------------------------------------------------------------------------------------ 
        // if it fits in shared memory do the work in shared memory
        //------------------------------------------------------------------------------------------------------------ 
        for (int j = k >> 1; j > TILE; j >>= 1) {
        //------------------------------------------------------------------------------------------------------------ 
        // fallback option if it does not fit in shared memory anymore
        //------------------------------------------------------------------------------------------------------------ 
            {
                int grid = (N + BLOCK - 1) / BLOCK;
                //printf("grid: %d ", grid);
                bitonic_step_for<<<grid, BLOCK>>>(dbuf, j, k, N);
                //(cudaGetLastError());
            }
        }
        {
            //int max_w = 2 * TILE;
            int grid = (N + (2*TILE) - 1) / (2*TILE);
            bitonic_shared<TILE><<<grid, BLOCK, ((2* TILE)+(2*TILE)/32)*sizeof(uint32_t)>>>(dbuf,k, N);
        }

    }
    //------------------------------------------------------------------------------------------------------------ 
    /**cudaDeviceSynchronize();   
    cudaEventRecord(stop_shared);
    cudaEventSynchronize(stop_shared);
    //------------------------------------------------------------------------------------------------------------ 
    cudaEventElapsedTime(&ms, start_shared, stop_shared);
    printf("GPU Time: %.3f ms\n", ms);
    cudaEventDestroy(start_shared);
    cudaEventDestroy(stop_shared);**/
    //------------------------------------------------------------------------------------------------------------
    //(cudaMemcpy(arr, dbuf, (size_t)N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    //(cudaFree(dbuf));
    cudaHostUnregister(arr);
    //------------------------------------------------------------------------------------------------------------
}
//------------------------------------------------------------------------------------------------------------
static inline void reverse_block_u32(uint32_t *a, int start, int K) {
    int i = start, j = start + K - 1;
    while (i < j) {
        uint32_t t = a[i]; a[i] = a[j]; a[j] = t;
        ++i; --j;
    }
}
//------------------------------------------------------------------------------------------------------------
static void make_alternating_runs(uint32_t *a, int N, int K) {
    for (int base = K; base + K <= N; base += 2*K) { 
        reverse_block_u32(a, base, K);
    }
}
//------------------------------------------------------------------------------------------------------------
extern "C" void hybrid_sort(uint32_t *arr, int N, int K_CPU_MAX) {


    int K = (K_CPU_MAX);
    printf("K: %d", K);
    printf("N: %d", N);
    bool use_cpu = false;
    if(K_CPU_MAX < 67108864)use_cpu = true;
    //------------------------------------------------------------------------------------------------------------
    clock_t start, end;
    double cpu_time_used;
    //------------------------------------------------------------------------------------------------------------
    struct timespec a, b; 
    clock_gettime(CLOCK_MONOTONIC,&a);
    //------------------------------------------------------------------------------------------------------------
    simd_mergesort_uint32_k(arr, N, K);
    clock_gettime(CLOCK_MONOTONIC,&b);
    double sec = (b.tv_sec-a.tv_sec) + (b.tv_nsec-a.tv_nsec)/1e9;
    printf("CPU time used sort: %.6f seconds\n", sec);
    //------------------------------------------------------------------------------------------------------------
    clock_gettime(CLOCK_MONOTONIC,&a);
    //if (src != arr) memcpy(arr, src, N * sizeof(uint32_t));
    make_alternating_runs(arr, N, K);
    clock_gettime(CLOCK_MONOTONIC,&b);
    sec = (b.tv_sec-a.tv_sec) + (b.tv_nsec-a.tv_nsec)/1e9;
    printf("CPU time used alternating: %.6f seconds\n", sec);
    //------------------------------------------------------------------------------------------------------------
    cudaEvent_t start_shared, stop_shared;
    float ms = 0.0f;
    cudaEventCreate(&start_shared);
    cudaEventCreate(&stop_shared);
    cudaEventRecord(start_shared);
    //cudaMemcpy(d_data, arr, N*4, cudaMemcpyHostToDevice);
    gpu_bitonic_sort_uint32_k(arr, N, K);
    //cudaMemcpy(arr, d_data, N*4, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();   
    cudaEventRecord(stop_shared);
    cudaEventSynchronize(stop_shared);
    //------------------------------------------------------------------------------------------------------------ 
    cudaEventElapsedTime(&ms, start_shared, stop_shared);
    printf("GPU Time: %.3f s\n", (ms / 1000));
    cudaEventDestroy(start_shared);
    cudaEventDestroy(stop_shared);
    //------------------------------------------------------------------------------------------------------------
}
//------------------------------------------------------------------------------------------------------------
extern "C" void hybrid_sort_chunk(uint32_t *arr, int N, int K_CPU_MAX) {
    int K = (K_CPU_MAX);
    printf("K: %d", K);
    printf("N: %d", N);
    bool use_cpu = false;
    if(K_CPU_MAX < 67108864)use_cpu = true;
    //------------------------------------------------------------------------------------------------------------
    clock_t start, end;
    double cpu_time_used;
    //------------------------------------------------------------------------------------------------------------
    struct timespec a, b; 
    clock_gettime(CLOCK_MONOTONIC,&a);
    //------------------------------------------------------------------------------------------------------------
    simd_mergesort_uint32_k(arr, N, K);
    clock_gettime(CLOCK_MONOTONIC,&b);
    double sec = (b.tv_sec-a.tv_sec) + (b.tv_nsec-a.tv_nsec)/1e9;
    printf("CPU time used sort: %.6f seconds\n", sec);
    //------------------------------------------------------------------------------------------------------------
    clock_gettime(CLOCK_MONOTONIC,&a);
    //if (src != arr) memcpy(arr, src, N * sizeof(uint32_t));
    make_alternating_runs(arr, N, K);
    clock_gettime(CLOCK_MONOTONIC,&b);
    sec = (b.tv_sec-a.tv_sec) + (b.tv_nsec-a.tv_nsec)/1e9;
    printf("CPU time used alternating: %.6f seconds\n", sec);
    //------------------------------------------------------------------------------------------------------------
    cudaEvent_t start_shared, stop_shared;
    float ms = 0.0f;
    cudaEventCreate(&start_shared);
    cudaEventCreate(&stop_shared);
    cudaEventRecord(start_shared);
    //cudaMemcpy(d_data, arr, N*4, cudaMemcpyHostToDevice);
    gpu_bitonic_sort_uint32_k(arr, N, K);
    //cudaMemcpy(arr, d_data, N*4, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();   
    cudaEventRecord(stop_shared);
    cudaEventSynchronize(stop_shared);
    //------------------------------------------------------------------------------------------------------------ 
    cudaEventElapsedTime(&ms, start_shared, stop_shared);
    printf("GPU Time: %.3f s\n", (ms / 1000));
    cudaEventDestroy(start_shared);
    cudaEventDestroy(stop_shared);
    //------------------------------------------------------------------------------------------------------------
}
//------------------------------------------------------------------------------------------------------------