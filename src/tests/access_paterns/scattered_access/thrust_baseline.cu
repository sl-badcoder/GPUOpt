//------------------------------------------------------------------------------------------------------------
#include <iostream>
#include <chrono>
#include <cstdint>
#include <cstdlib>

#include <cuda_runtime.h>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
//------------------------------------------------------------------------------------------------------------
using std::cout;
using std::endl;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
//------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " log2(N)\n";
        return 1;
    }

    size_t N = 1ULL << atol(argv[1]);
    cout << "N = " << N << endl;

    uint32_t *data = nullptr;
    cudaError_t err = cudaMallocManaged(&data, N * sizeof(uint32_t), cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocManaged failed: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    thrust::default_random_engine rng(1337);
    thrust::uniform_int_distribution<uint32_t> dist;

    for (size_t i = 0; i < N; ++i) {
        data[i] = dist(rng);
    }
    auto start = high_resolution_clock::now();
    int device = 0;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(data, N * sizeof(uint32_t), device, 0);

    thrust::sort(thrust::device, data, data + N);

    cudaDeviceSynchronize(); 
    auto end = high_resolution_clock::now();

    duration<double> elapsed = duration_cast<duration<double>>(end - start);
    cout << "Sort time: " << elapsed.count() << " s" << endl;


    cudaFree(data);

    int dev = 0;
    cudaGetDevice(&dev);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    cout << prop.sharedMemPerBlock << endl;


    return 0;
}
//------------------------------------------------------------------------------------------------------------
