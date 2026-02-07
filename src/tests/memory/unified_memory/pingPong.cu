
//------------------------------------------------------------------------------------------------------------
#include <iostream>
#include <chrono>
#include <vector>
#include "../../../../include/core/helper.h" // maybe change complete path later
#include "../../../../include/cost_model/gpu_usage.h"
#include "../../../../include/cost_model/cpu_usage.h"
//------------------------------------------------------------------------------------------------------------
#include "cuda_runtime.h"
#include <sys/mman.h>
//------------------------------------------------------------------------------------------------------------
using std::cout;
using std::endl;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
//------------------------------------------------------------------------------------------------------------
__global__ void add1(uint32_t* arr, size_t N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N)arr[i] += 1;
}
//------------------------------------------------------------------------------------------------------------
void arrayTest(uint32_t* tmp, size_t N){
    //for(int i=0;i<10;i++){
        for(size_t i =0;i<N;i+=1){
            tmp[i]++;
        }
    //}
}    
size_t L3CACHE = 33554432;
//------------------------------------------------------------------------------------------------------------
void warmup_cache(){
    std::vector<char> val(L3CACHE, 0);
    for(int i=0;i<L3CACHE;i++){
        val[i] += 7;
    }
}
size_t Gib = (size_t)1024*1024*1024;
//------------------------------------------------------------------------------------------------------------
size_t actualPrefetchBytes(size_t N, size_t free_bytes){
    size_t buffer = 512ULL * 1024 * 1024; // 512 MiB safety buffer
    size_t safe_prefetch_bytes = (free_bytes > buffer) ? (free_bytes - buffer) : 0;

    size_t total_needed_bytes = N;

    size_t actual_prefetch_bytes = std::min(total_needed_bytes, safe_prefetch_bytes);
    return actual_prefetch_bytes;
}
//------------------------------------------------------------------------------------------------------------
void pingPongAccess(size_t N){
    N = (size_t)N/(size_t)4;
    // cudaStream_t stream;
    // cudaStreamCreate(&stream);
    // cudaStream_t s2;
    // cudaStreamCreate(&s2);
    uint32_t* tmp;
    CHECK_CUDA(cudaMallocManaged((void**)&tmp, N * sizeof(uint32_t), cudaMemAttachGlobal)); 
    for(size_t i =0;i<N;i++){
        tmp[i] = 1;
    }
    auto start = high_resolution_clock::now();

    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;   
    loc.id   = device;  
    int grid = (int)((N+255)/256);
    size_t actual_prefetch_bytes = actualPrefetchBytes(N, getGPUFreeMemory());
    // CHECK_CUDA(cudaMemAdvise((void*)tmp, actual_prefetch_bytes, cudaMemAdviseSetAccessedBy, loc.id));
    // CHECK_CUDA(cudaMemAdvise((void*)tmp, actual_prefetch_bytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));
    // CHECK_CUDA(cudaMemAdvise(tmp, N, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    // CHECK_CUDA(cudaMemAdvise(tmp, N, cudaMemAdviseSetAccessedBy, device));
    // RAM -> VRAM
    //  size_t actual_prefetch_bytes = actualPrefetchBytes(N, getGPUFreeMemory());
   // if (actual_prefetch_bytes > 0)CHECK_CUDA(cudaMemPrefetchAsync(tmp, actual_prefetch_bytes, device, s2));
    //cudaStreamSynchronize(stream);
    //CHECK_CUDA(cudaDeviceSynchronize());    
    add1<<< grid, 256>>>(tmp, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    // VRAM -> RAM
    //  actual_prefetch_bytes = actualPrefetchBytes(N, getGPUFreeMemory());
    //if (actual_prefetch_bytes > 0)CHECK_CUDA(cudaMemPrefetchAsync(tmp, actual_prefetch_bytes, cudaCpuDeviceId, s2));
    //cudaStreamSynchronize(stream);
    // CHECK_CUDA(cudaFree(tmp));
    // CHECK_CUDA(cudaMallocManaged((void**)&tmp, N * sizeof(uint32_t), cudaMemAttachGlobal)); 

    arrayTest(tmp, N);
    // // RAM -> VRAM
    // actual_prefetch_bytes = actualPrefetchBytes(N, getGPUFreeMemory());
//    if (actual_prefetch_bytes > 0)CHECK_CUDA(cudaMemPrefetchAsync(tmp, actual_prefetch_bytes, device, s2));    
   // cudaStreamSynchronize(stream);
    // CHECK_CUDA(cudaFree(tmp));
    // CHECK_CUDA(cudaMallocManaged((void**)&tmp, N * sizeof(uint32_t), cudaMemAttachGlobal)); 

    add1<<< grid, 256>>>(tmp, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    // VRAM -> RAM
    // actual_prefetch_bytes = actualPrefetchBytes(N, getGPUFreeMemory());
    //if (actual_prefetch_bytes > 0)CHECK_CUDA(cudaMemPrefetchAsync(tmp, actual_prefetch_bytes, cudaCpuDeviceId, s2));

    
    //cudaStreamSynchronize(stream);
    
    // CHECK_CUDA(cudaFree(tmp));
    // CHECK_CUDA(cudaMallocManaged((void**)&tmp, N * sizeof(uint32_t), cudaMemAttachGlobal)); 

    
    arrayTest(tmp, N);


    // cudaStreamSynchronize(stream);
    // cudaStreamDestroy(stream);
    // cudaStreamSynchronize(s2);
    // cudaStreamDestroy(s2);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = high_resolution_clock::now();
    CHECK_CUDA(cudaFree(tmp));
    CHECK_CUDA(cudaGetLastError());
    auto duration = end - start;
    cout << "Time taken pingPong unified: " << duration_cast<std::chrono::milliseconds>(duration).count() << " ms" << endl;
}


int main(){
    size_t MiB = (size_t)1024*1024;
    size_t Gib = (size_t)1024*1024*1024;
    std::vector<size_t> values = {1, 32, 1024, 1048576};
    //values = {1024};
    std::vector<size_t> sizes = {(size_t)1 * Gib,(size_t) 2 * Gib, (size_t)4 * Gib, (size_t)8 * Gib, (size_t)12 * Gib, (size_t)16 * Gib, (size_t) 20 * Gib,(size_t) 24 * Gib, (size_t)26 * Gib, (size_t)28 * Gib};
    // sizes = {(size_t)26 * Gib, (size_t)28 * Gib};
    sizes = {20 * Gib};
    for(auto size : sizes){
        warmup_cache();
        pingPongAccess(size);
    }


    return EXIT_SUCCESS;
}