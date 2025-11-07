// shared_mem_sort.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <limits>

__global__ void bitonic_sort_shared(float* data, int N)
{
    extern __shared__ float s[];
    const int tid = threadIdx.x;
    const int n   = blockDim.x; 
    float val = INFINITY;                  
    if (tid < N) val = data[tid];
    s[tid] = val;
    __syncthreads();

    // Bitonic sort network (ascending)
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                bool ascending = ((tid & k) == 0);
                float a = s[tid];
                float b = s[ixj];
                // Swap if out of order for the desired direction
                if ((a > b) == ascending) {
                    s[tid] = b;
                    s[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    // Write back
    if (tid < N) data[tid] = s[tid];
}

// ---- Host helpers ----
static int next_pow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

extern "C" void gpu_bitonic_sort_uint32_shared(uint32_t *arr, int N){
    if (N <= 0 || N > 1024) {
        fprintf(stderr, "Please choose 1 <= N <= 1024 for this simple demo.\n");
        return 1;
    }

    // Create some data on host
    std::vector<float> h(N);
    for (int i = 0; i < N; ++i) {
        h[i] = static_cast<float>(rand()) / RAND_MAX; // random [0,1)
    }

    // Allocate device buffer and copy data
    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch: one block, T threads = next power of two, dynamic shared mem = T*sizeof(float)
    int T = next_pow2(N);
    // Safety: cap at device maximum threads per block just in case (this demo expects N<=1024)
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (T > prop.maxThreadsPerBlock) {
        fprintf(stderr, "Device max threads per block = %d, requested = %d\n", prop.maxThreadsPerBlock, T);
        return 1;
    }

    bitonic_sort_shared<<<1, T, T * sizeof(float)>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back and print a few values
    CUDA_CHECK(cudaMemcpy(h.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
}
