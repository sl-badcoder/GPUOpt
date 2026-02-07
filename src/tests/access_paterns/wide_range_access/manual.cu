//---------------------------------------------------------------------------------------
#include "cuda_runtime.h"
#include "../../../../include/core/helper.h" // maybe change complete path later
#include "../../../../include/cost_model/gpu_usage.h"
#include "../../../../include/cost_model/cpu_usage.h"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <fstream>
using std::cout;
using std::endl;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
//---------------------------------------------------------------------------------------
__global__ void wide_range_access_kernel(uint32_t* data, size_t N, int width) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N - width) {
        int sum = data[i] + data[i + width];
        
        data[i] = sum * 10;
    }
}
void arrayTest(uint32_t* tmp, size_t N, size_t width){
    //for(int i=0;i<10;i++){
        for(size_t i =0;i<N-width;i+=1){
            int sum = tmp[i] + tmp[i+width];
            sum *= 10;
        }
    //}
}   
void log_to_csv(size_t size_bytes, size_t width, std::string mode, long long time_ms) {
    std::ofstream file;
    file.open("experiment_results.csv", std::ios_base::app);
    if (file.is_open()) {
        file << size_bytes << "," << width << "," << mode << "," << time_ms << "\n";
        file.close();
    }
}
//------------------------------------------------------------------------------------------------------------
size_t L3CACHE = 33554432;
//------------------------------------------------------------------------------------------------------------
void warmup_cache(){
    std::vector<char> val(L3CACHE, 0);
    for(int i=0;i<L3CACHE;i++){
        val[i] += 1;
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
    auto start = high_resolution_clock::now();
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
    auto end = high_resolution_clock::now();
    CHECK_CUDA(cudaFree(data));
    long long duration = duration_cast<std::chrono::milliseconds>(end - start).count();
    cout << "Time taken wo prefetch wo memadvise: " << duration << " ms" << endl;
    log_to_csv(N, width, "woPrefetchwoMemadvise", duration);
}
//---------------------------------------------------------------------------------------
void wide_stride_access_wo_prefetch(size_t N, size_t width){
    uint32_t* data = nullptr;
    CHECK_CUDA(cudaMallocManaged(&data, N * sizeof(uint32_t), cudaMemAttachGlobal)); // allocate unified memory
    auto start = high_resolution_clock::now();

    int threadsPerBlock = 256;
    int blocksPerGrid = (N - width + threadsPerBlock - 1) / threadsPerBlock;
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;   
    loc.id   = device;  
    int grid = (int)((N+255)/256);
    CHECK_CUDA(cudaMemAdvise((void*)data, N * sizeof(uint32_t), cudaMemAdviseSetAccessedBy, loc.id));
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
    
    auto end = high_resolution_clock::now();
    long long duration = duration_cast<std::chrono::milliseconds>(end - start).count();
    cout << "Time taken wo prefetch: " << duration << " ms" << endl;
    log_to_csv(N, width, "woPrefetch", duration);
    CHECK_CUDA(cudaFree(data));
}
//---------------------------------------------------------------------------------------
void wide_stride_access_wo_memAdvise(size_t N, size_t width){
    uint32_t* data = nullptr;
    CHECK_CUDA(cudaMallocManaged(&data, N * sizeof(uint32_t), cudaMemAttachGlobal)); // allocate unified memory
    auto start = high_resolution_clock::now();

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
    auto end = high_resolution_clock::now();
    long long duration = duration_cast<std::chrono::milliseconds>(end - start).count();
    cout << "Time taken wo memAdvise: " << duration << " ms" << endl;
    log_to_csv(N, width, "womemAdvise", duration);
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaStreamDestroy(s2));
    CHECK_CUDA(cudaFree(data));

}
size_t MiB = (size_t)1024*1024;
size_t Gib = (size_t)1024*1024*1024;
void pingPongAccess(size_t N, size_t width){

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaStream_t s2;
    cudaStreamCreate(&s2);
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
   //  printf("%zu\n", getGPUFreeMemory);
   // if (actual_prefetch_bytes > 0)CHECK_CUDA(cudaMemPrefetchAsync(tmp,actual_prefetch_bytes, device, s2));
    //cudaStreamSynchronize(stream);
    //CHECK_CUDA(cudaDeviceSynchronize());    
    wide_range_access_kernel<<< grid, 256>>>(tmp, N, width);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    // VRAM -> RAM

   //  if (actual_prefetch_bytes > 0)CHECK_CUDA(cudaMemPrefetchAsync(tmp, actual_prefetch_bytes, cudaCpuDeviceId, 0));
    //cudaStreamSynchronize(stream);
    // CHECK_CUDA(cudaFree(tmp));
    // CHECK_CUDA(cudaMallocManaged((void**)&tmp, N, cudaMemAttachGlobal)); 

    arrayTest(tmp, N, width);
    // RAM -> VRAM
  //  if (actual_prefetch_bytes > 0)CHECK_CUDA(cudaMemPrefetchAsync(tmp,actual_prefetch_bytes, device, 0));
    
    cudaStreamSynchronize(stream);
    // CHECK_CUDA(cudaFree(tmp));
    // CHECK_CUDA(cudaMallocManaged((void**)&tmp, N, cudaMemAttachGlobal)); 

    wide_range_access_kernel<<< grid, 256>>>(tmp, N, width);
    CHECK_CUDA(cudaDeviceSynchronize());
    // VRAM -> RAM
   //  if (actual_prefetch_bytes > 0)CHECK_CUDA(cudaMemPrefetchAsync(tmp, actual_prefetch_bytes, cudaCpuDeviceId, 0));
    
    cudaStreamSynchronize(stream);
    
    // CHECK_CUDA(cudaFree(tmp));
    // CHECK_CUDA(cudaMallocManaged((void**)&tmp, N, cudaMemAttachGlobal)); 

    
    arrayTest(tmp, N, width);


    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaStreamSynchronize(s2);
    cudaStreamDestroy(s2);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = high_resolution_clock::now();
    CHECK_CUDA(cudaFree(tmp));
    auto duration = end - start;
    cout << "Time taken pingPong unified: " << duration_cast<std::chrono::milliseconds>(duration).count() << " ms" << endl;
}

//---------------------------------------------------------------------------------------
int main(){
    // check different strides against different memory sizes
    // size_t MiB = (size_t)1024*1024;
    // size_t Gib = (size_t)1024*1024*1024;
     std::vector<size_t> values = {1, 32, 1024, 1048576};
    // //values = {1024};
    std::vector<size_t> sizes = {(size_t)1 * Gib,(size_t) 2 * Gib, (size_t)4 * Gib, (size_t)8 * Gib, (size_t)12 * Gib, (size_t)16 * Gib, (size_t) 20 * Gib,(size_t) 24 * Gib, (size_t)26 * Gib, (size_t)28 * Gib};
    // //sizes = {(size_t)20 * Gib};
    // for(int i=0; i < 20; i++){
         for(auto width : values){
             for(auto sz: sizes){
                 sz = (size_t)sz/(size_t)4;
                 cout << "sizes: " << sz * 4 << " width: " << width << endl;
    //             warmup_cache();
    //             wide_stride_access_wo_prefetch_wo_memAdvise(sz,width );
    //             warmup_cache();
    //             wide_stride_access_wo_prefetch(sz, width);
    //             warmup_cache();
                 wide_stride_access_wo_memAdvise(sz, width);
            }
         }
    // }
    // sizes = {20 * Gib};
    // for(auto sz : sizes)pingPongAccess((size_t)sz/(size_t)4, 1048576);
    return 0;
}