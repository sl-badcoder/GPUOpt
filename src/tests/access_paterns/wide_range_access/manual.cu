//---------------------------------------------------------------------------------------
#include "cuda_runtime.h"
#include "../../../../include/core/helper.h" // maybe change complete path later
#include "../../../../include/cost_model/gpu_usage.h"
#include "../../../../include/cost_model/cpu_usage.h"
#include <iostream>
#include <stdio.h>
//---------------------------------------------------------------------------------------
__global__ void wide_range_access_kernel(uint32_t* data, size_t N, int width) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N - width) {
        int sum = data[i] + data[i + width];
        
        data[i] = sum * 10;
    }
}

size_t actualPrefetchBytes(size_t N, size_t free_bytes){
    size_t buffer = 512ULL * 1024 * 1024; // 512 MiB safety buffer
    size_t safe_prefetch_bytes = (free_bytes > buffer) ? (free_bytes - buffer) : 0;

    size_t total_needed_bytes = N * sizeof(uint32_t);

    size_t actual_prefetch_bytes = std::min(total_needed_bytes, safe_prefetch_bytes);
    return actual_prefetch_bytes;
}
//---------------------------------------------------------------------------------------
void wide_stride_access_wo_prefetch_wo_memAdvise(size_t N, size_t width){
    uint32_t* data = nullptr;
    CHECK_CUDA(cudaMallocManaged(&data, N * sizeof(uint32_t), cudaMemAttachGlobal)); // allocate unified memory
    int threadsPerBlock = 256;
    int blocksPerGrid = (N - width + threadsPerBlock - 1) / threadsPerBlock;

    for(size_t i = 0; i < N - width; i++){
        int sum = data[i] + data[i+width];
        sum *= 10;
    }

    wide_range_access_kernel<<<blocksPerGrid, threadsPerBlock>>>(data, N, width); // GPU wide-range access
    CHECK_CUDA(cudaDeviceSynchronize());
    int tst = 0;
    for(size_t i = 0; i< N; i++){
        tst += data[i];
    }

    int a = tst + 20;
    CHECK_CUDA(cudaFree(data));
}
//---------------------------------------------------------------------------------------
void wide_stride_access_wo_prefetch(size_t N, size_t width){
    uint32_t* data = nullptr;
    CHECK_CUDA(cudaMallocManaged(&data, N * sizeof(uint32_t), cudaMemAttachGlobal)); // allocate unified memory
    int threadsPerBlock = 256;
    int blocksPerGrid = (N - width + threadsPerBlock - 1) / threadsPerBlock;
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;   
    loc.id   = device;  
    int grid = (int)((N+255)/256);
    CHECK_CUDA(cudaMemAdvise((void*)data, N * sizeof(uint32_t), cudaMemAdviseSetAccessedBy, 0));
    CHECK_CUDA(cudaMemAdvise((void*)data, N* sizeof(uint32_t), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));

    for(size_t i = 0; i < N - width; i++){
        int sum = data[i] + data[i+width];
        sum *= 10;
    }
    wide_range_access_kernel<<<blocksPerGrid, threadsPerBlock>>>(data, N, width); // GPU wide-range access
    CHECK_CUDA(cudaDeviceSynchronize());
    int tst = 0;
    for(size_t i = 0; i< N; i++){
        tst += data[i];
    }

    int a = tst + 20;

    CHECK_CUDA(cudaFree(data));
}
//---------------------------------------------------------------------------------------
void wide_stride_access_wo_memAdvise(size_t N, size_t width){
    uint32_t* data = nullptr;
    CHECK_CUDA(cudaMallocManaged(&data, N * sizeof(uint32_t), cudaMemAttachGlobal)); // allocate unified memory
    int threadsPerBlock = 256;
    int blocksPerGrid = (N - width + threadsPerBlock - 1) / threadsPerBlock;
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaStream_t s2;
    cudaStreamCreate(&s2);
    cudaMemLocation loc{};
    
    loc.type = cudaMemLocationTypeDevice;   
    loc.id   = device;  
    int grid = (int)((N+255)/256);
   //CHECK_CUDA(cudaMemAdvise((void*)data, N, cudaMemAdviseSetAccessedBy, loc.id));
    //CHECK_CUDA(cudaMemAdvise((void*)data, N, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));
    // Pinning Unified Memory With this command
    CHECK_CUDA(cudaMemAdvise(data, N * sizeof(uint32_t), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CHECK_CUDA(cudaMemAdvise(data, N * sizeof(uint32_t), cudaMemAdviseSetAccessedBy, device));
    for(size_t i = 0; i < N - width; i++){
        int sum = data[i] + data[i+width];
        sum *= 10;
    }
    size_t actual_prefetch_bytes = actualPrefetchBytes(N, getGPUFreeMemory());
    if (actual_prefetch_bytes > 0) {
        CHECK_CUDA(cudaMemPrefetchAsync(data, actual_prefetch_bytes, device, s2));
    }
    wide_range_access_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(data, N, width); // GPU wide-range access
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    if (actual_prefetch_bytes > 0) {
        CHECK_CUDA(cudaMemPrefetchAsync(data, actual_prefetch_bytes, cudaCpuDeviceId, s2));
    }
    
    int tst = 0;
    for(size_t i = 0; i< N; i++){
        tst += data[i];
    }

    int a = tst + 20;
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaStreamDestroy(s2));
    CHECK_CUDA(cudaFree(data));

}
//---------------------------------------------------------------------------------------
int main(){
    wide_stride_access_wo_memAdvise((size_t)1024 * 1024 * 1024 * 5, 10000);
    return 0;
}