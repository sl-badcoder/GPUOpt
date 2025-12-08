//------------------------------------------------------------------------------------------------------------
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
//------------------------------------------------------------------------------------------------------------
#include "gpu/bitonic_sort.h"
#include "cpu/bitonic_simd_merge.h"
//------------------------------------------------------------------------------------------------------------
#define _POSIX_C_SOURCE 199309L
#define CHECK_CUDA(call) do {                                         \
  cudaError_t _e = (call);                                            \
  if (_e != cudaSuccess) {                                            \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
            cudaGetErrorString(_e));                                  \
    exit(1);                                                          \
  }                                                                   \
} while (0)
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
//------------------------------------------------------------------------------------------------------------
__global__ void reverse_kernel_uint32(uint32_t *arr, size_t N)
{
    // Each thread swaps one pair (i, N-1-i)
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t i   = tid;
    size_t j   = N - 1 - i;

    if (i < j) {
        uint32_t a = arr[i];
        uint32_t b = arr[j];
        arr[i] = b;
        arr[j] = a;
    }
}
//------------------------------------------------------------------------------------------------------------
template<size_t TILE>
__global__ void bitonic_block_sort(uint32_t *d, size_t n) {
    extern __shared__ uint32_t s[]; // define array in shared memory 
    size_t base = blockIdx.x * TILE;
    size_t tid  = threadIdx.x;
    //------------------------------------------------------------------------------------------------------------
    auto smem_idx = [&](size_t i){ return i + (i >> 5); };
    //------------------------------------------------------------------------------------------------------------
    for (size_t i = tid; i < TILE; i += blockDim.x) {
        size_t gi = base + i;
        s[smem_idx(i)] = (gi < n) ? d[gi] : 0xFFFFFFFFu;
    }
    __syncthreads();
    //------------------------------------------------------------------------------------------------------------
    for (size_t k = 2; k <= TILE; k <<= 1) {
        for (size_t j = k >> 1; j > 0; j >>= 1) {
            for (size_t i = tid; i < TILE; i += blockDim.x) {
                size_t ixj = i ^ j;
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
        for(size_t i = tid; i < TILE/2; i+=(size_t)blockDim.x){
            size_t a = i;
            size_t b = TILE - 1 - i;
            uint32_t va = s[smem_idx(a)];
            uint32_t vb = s[smem_idx(b)];
            s[smem_idx(a)] = vb;
            s[smem_idx(b)] = va;
        }
        __syncthreads();
    }
    //------------------------------------------------------------------------------------------------------------
    for (size_t i = tid; i < TILE; i +=(size_t) blockDim.x) {
        size_t gi = base + i;
        if (gi < n) d[gi] = s[smem_idx(i)];
    }
    //------------------------------------------------------------------------------------------------------------
}
//------------------------------------------------------------------------------------------------------------
template<size_t TILE>
__global__ void bitonic_shared(uint32_t* data, size_t k, size_t n){
    extern __shared__ uint32_t s[]; // shared memory array
    size_t max_w = 2 * TILE;
    size_t base = blockIdx.x * (2 * TILE);
    size_t tid = threadIdx.x;

    // propely access the shared memory array on the GPU
    auto smem_idx = [&](size_t i){
        return i + (i>>5);
    };

    // load data from data array
    for(size_t i = tid; i < max_w; i+=(size_t)blockDim.x){
        size_t gi = base + i;
        s[smem_idx(i)] = (gi < n) ? data[gi] : 0xFFFFFFFFu;
    }
    __syncthreads();
    //------------------------------------------------------------------------------------------------------------
    // normal bitonic sort logic updated to global 
    //------------------------------------------------------------------------------------------------------------
    for(size_t j = min(TILE, k >> 1); j >0 ; j >>=1){
        for(size_t i = tid; i < max_w; i+=(size_t)blockDim.x){
            size_t gi = base + i;
            size_t g_ixj = gi ^ j;  
            if (g_ixj >= base && g_ixj < (base + max_w) && g_ixj > gi) {
                size_t l_i = i;
                size_t l_ixj = g_ixj - base;
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
    for(size_t i = tid; i < max_w; i+=(size_t)blockDim.x){
        size_t gi = base + i;
        if(gi<n) data[gi] = s[smem_idx(i)];
    }
}
//------------------------------------------------------------------------------------------------------------
__global__ void bitonic_step(uint32_t *data, size_t j, size_t k, size_t n) {
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) return;

    size_t ixj = i ^ j;  
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
//------------------------------------------------------------------------------------------------------------
// like bitonic step but with for loop. 
__global__ void bitonic_step_for(uint32_t *data, size_t j, size_t k, size_t n) {
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t stride = (size_t)gridDim.x * (size_t)blockDim.x;

    for(; i <n; i+= stride){
        size_t ixj = i ^ j;  
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
extern "C" void gpu_bitonic_sort_uint32(uint32_t *arr, size_t N) {
    if (N <= 1) return;
    //------------------------------------------------------------------------------------------------------------
    // allocate the buffer array
    //------------------------------------------------------------------------------------------------------------
    uint32_t *dbuf = NULL;
    CHECK_CUDA(cudaMalloc((void**)&dbuf, (size_t)N * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(dbuf, arr, (size_t)N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    //------------------------------------------------------------------------------------------------------------
    /**cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);**/
    //------------------------------------------------------------------------------------------------------------
    // define block size
    const size_t BLOCK = 1024;
    const size_t TILE = 8 * BLOCK;
    dim3 block(BLOCK);
    size_t shmem_block = (TILE + TILE/32) * sizeof(uint32_t);
    //------------------------------------------------------------------------------------------------------------
    //cudaEventRecord(start);
    size_t grid_b = (N + TILE - 1)/ TILE;
    CHECK_CUDA(cudaGetLastError());
    bitonic_block_sort<TILE><<<grid_b, BLOCK, shmem_block>>>(dbuf, N);
    CHECK_CUDA(cudaGetLastError());
    for (size_t k = 2 * TILE; k <= N; k <<= 1) {
        //------------------------------------------------------------------------------------------------------------ 
        // if it fits in shared memory do the work in shared memory
        //------------------------------------------------------------------------------------------------------------ 
        for (size_t j = k >> 1; j > TILE; j >>= 1) {
        //------------------------------------------------------------------------------------------------------------ 
        // fallback option if it does not fit in shared memory anymore
        //------------------------------------------------------------------------------------------------------------ 
            {
                size_t grid = (N + BLOCK - 1) / BLOCK;
                bitonic_step_for<<<grid, BLOCK>>>(dbuf, j, k, N);
                CHECK_CUDA(cudaGetLastError());
            }
        }
        {
            size_t grid = (N + (2*TILE) - 1) / (2*TILE);
            bitonic_shared<TILE><<<grid, BLOCK, ((2* TILE)+(2*TILE)/32)*sizeof(uint32_t)>>>(dbuf,k, N);
            CHECK_CUDA(cudaGetLastError());
        }

    }
    //------------------------------------------------------------------------------------------------------------
    CHECK_CUDA(cudaMemcpy(arr, dbuf, (size_t)N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(dbuf));
    //------------------------------------------------------------------------------------------------------------
}


extern "C" void gpu_bitonic_sort_uint32_k(uint32_t *arr, size_t N, int k_start, bool MAPPED) {
    if (N <= 1) return;
    //------------------------------------------------------------------------------------------------------------
    // check if we map memory or not
    //------------------------------------------------------------------------------------------------------------
    uint32_t *dbuf = NULL;
    if(MAPPED){
        CHECK_CUDA(cudaHostRegister(arr, N * (sizeof(uint32_t)), cudaHostRegisterMapped));
        CHECK_CUDA(cudaHostGetDevicePointer(&dbuf, arr, 0)); 
    }else{
        cudaStream_t s;
        CHECK_CUDA(cudaStreamCreate(&s));
        CHECK_CUDA(cudaMalloc((void**)&dbuf, (size_t)N * sizeof(uint32_t)));
        CHECK_CUDA(cudaMemcpyAsync(dbuf, arr, (size_t)N * sizeof(uint32_t), cudaMemcpyHostToDevice, s));
    }
    //------------------------------------------------------------------------------------------------------------
    // define block size
    const size_t BLOCK = 1024;
    const size_t TILE = 4 * BLOCK;
    dim3 block(BLOCK);
    size_t shmem_block = (TILE + TILE/32) * sizeof(uint32_t);
    //------------------------------------------------------------------------------------------------------------
    size_t grid_b = (N + TILE - 1)/ TILE;
    //------------------------------------------------------------------------------------------------------------ 
    for (size_t k = 2 * k_start; k <= N; k <<= 1) {
        //------------------------------------------------------------------------------------------------------------ 
        for (size_t j = k >> 1; j > TILE; j >>= 1) {
        //------------------------------------------------------------------------------------------------------------ 
            {
                size_t grid = (N + BLOCK - 1) / BLOCK;
                bitonic_step_for<<<grid, BLOCK>>>(dbuf, j, k, N);
            }
        }
        {
            size_t grid = (N + (2*TILE) - 1) / (2*TILE);
            bitonic_shared<TILE><<<grid, BLOCK, ((2* TILE)+(2*TILE)/32)*sizeof(uint32_t)>>>(dbuf,k, N);
        }

    }
    //------------------------------------------------------------------------------------------------------------
    if(MAPPED){
        cudaHostUnregister(arr);
    }else{
        cudaStream_t s;
        CHECK_CUDA(cudaStreamCreate(&s));
        CHECK_CUDA(cudaMemcpyAsync(arr, dbuf, (size_t)N * sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
        CHECK_CUDA(cudaFree(dbuf));
    }
    //------------------------------------------------------------------------------------------------------------
}
//------------------------------------------------------------------------------------------------------------
extern "C" void gpu_bitonic_sort_uint32_chunk(uint32_t *dbuf, size_t N, cudaStream_t s) {
    if (N <= 1) return;

    //------------------------------------------------------------------------------------------------------------
    // define block size
    const size_t BLOCK = 1024;
    const size_t TILE = 4 * BLOCK;
    dim3 block(BLOCK);
    size_t shmem_block = (TILE + TILE/32) * sizeof(uint32_t);
    //------------------------------------------------------------------------------------------------------------
    size_t grid_b = (N + TILE - 1)/ TILE;
    bitonic_block_sort<TILE><<<grid_b, BLOCK, shmem_block, s>>>(dbuf, N);
    //------------------------------------------------------------------------------------------------------------
    for (size_t k = 2 * TILE; k <= N; k <<= 1) {
        //------------------------------------------------------------------------------------------------------------
        for (size_t j = k >> 1; j > TILE; j >>= 1) {
        //------------------------------------------------------------------------------------------------------------
            {
                size_t grid = (N + BLOCK - 1) / BLOCK;
                bitonic_step_for<<<grid, BLOCK, 0, s>>>(dbuf, j, k, N);
            }
        }
        {
            size_t grid = (N + (2*TILE) - 1) / (2*TILE);
            bitonic_shared<TILE><<<grid, BLOCK, ((2* TILE)+(2*TILE)/32)*sizeof(uint32_t), s>>>(dbuf,k, N);
        }

    }
    //------------------------------------------------------------------------------------------------------------
}
//------------------------------------------------------------------------------------------------------------
extern "C" void gpu_bitonic_sort_uint32_k_un(uint32_t *arr, size_t N, size_t k_start){
    printf("[NUMBER] OF ELEMENTS: %zu\n", N);
    //N = N / (size_t)4;
    if (N <= 1) return;
    //------------------------------------------------------------------------------------------------------------
    // check if we map memory or not
    //------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------
    // define block size
    const size_t BLOCK = 1024;
    const size_t TILE = 4 * BLOCK;
    dim3 block(BLOCK);
    size_t shmem_block = (TILE + TILE/32) *(size_t) sizeof(uint32_t);
    //------------------------------------------------------------------------------------------------------------
    size_t grid_b = (N + TILE - 1)/ TILE;
    //------------------------------------------------------------------------------------------------------------ 
    for (size_t k = 2 * k_start; k <= N; k <<= 1LL) {
        //------------------------------------------------------------------------------------------------------------ 
        for (size_t j = k >> 1; j > TILE; j >>= 1LL) {
        //------------------------------------------------------------------------------------------------------------ 
            {
                size_t grid = (N + BLOCK - 1) / BLOCK;
                bitonic_step_for<<<grid, BLOCK>>>(arr, j, k, N);
                CHECK_CUDA(cudaGetLastError());

            }
        }
        {
            size_t grid = (N + (2*TILE) - 1) / (2*TILE);
            bitonic_shared<TILE><<<grid, BLOCK, ((2* TILE)+(2*TILE)/32)*sizeof(uint32_t)>>>(arr,k, N);
            CHECK_CUDA(cudaGetLastError());
        }
        //printf("k:%zu\n", k);
    }
    //printf("END\n");
    //------------------------------------------------------------------------------------------------------------
}
//------------------------------------------------------------------------------------------------------------
// reverse the way how we run gpu_bitonic_sort
extern "C" void gpu_bitonic_sort_uint32_k_un_b(uint32_t *arr, size_t N, size_t k_start, bool asc)
{
    if (N <= 1) return;
    gpu_bitonic_sort_uint32_k_un(arr, N, k_start);
    if (!asc) {
        const int BLOCK = 1024;
        size_t pairs = N / 2;
        size_t grid  = (pairs + BLOCK - 1) / BLOCK;

        reverse_kernel_uint32<<<grid, BLOCK>>>(arr, N);
        CHECK_CUDA(cudaGetLastError());
    }
}
//------------------------------------------------------------------------------------------------------------