#include <iostream>
#include <chrono>
#include <vector>

#include "cuda_runtime.h"
#include <sys/mman.h>


using std::cout;
using std::endl;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

#define CHECK_CUDA(call) do {                                         \
  cudaError_t _e = (call);                                            \
  if (_e != cudaSuccess) {                                            \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
            cudaGetErrorString(_e));                                  \
    exit(1);                                                          \
  }                                                                   \
} while (0)

extern "C" void add1_simd(char *data, size_t N);


void test_cpu(size_t N){
    void *t = mmap(NULL, N, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS , -1, 0);
    char *arr = (char*)t;
    if(t == MAP_FAILED){
        printf("allocating MMAP failed");
        exit(1);
    }
    for(size_t i =0; i<N; i++){
        arr[i] = 0;
    }
    auto start = high_resolution_clock::now();

    for(size_t i =0; i< N; i++){
        arr[i] += 1;
    }

    auto end = high_resolution_clock::now();
    
    if(munmap(t, N) == -1){
        perror("munmap");
    }
    
    duration<double> duration = end - start;
    cout << "Time taken MMAP: " << duration_cast<std::chrono::milliseconds>(duration).count() << " milliseconds" << endl;
    
}

__global__ void add1(char* arr, size_t N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N)arr[i] += 1;
}

void test_cudaMallocHost(size_t N){
    char* data;

    CHECK_CUDA(cudaMallocHost((void**)&data, N)); 
    for(size_t i =0;i<N;i++){
        data[i] = 0;
    }
    char* dbuf = NULL;
    CHECK_CUDA(cudaMalloc((void**)&dbuf, (size_t)N));

    cudaStream_t s;
    CHECK_CUDA(cudaStreamCreate(&s));
    CHECK_CUDA(cudaMemcpyAsync(dbuf, data, (size_t)N , cudaMemcpyHostToDevice, s));
    int grid = (int)((N+255)/256);
    add1<<< grid, 256, 0, s>>>(dbuf, N);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpyAsync(data, dbuf, (size_t)N , cudaMemcpyDeviceToHost, s));
    
    CHECK_CUDA(cudaStreamSynchronize(s));
    CHECK_CUDA(cudaFree(dbuf));
    CHECK_CUDA(cudaStreamDestroy(s));
    auto start = high_resolution_clock::now();
    
    add1_simd(data, N);
    auto end = high_resolution_clock::now();

    CHECK_CUDA(cudaFreeHost(data));
    
    auto duration = end - start;
    cout << "Time taken cudaMallocHost: " << duration_cast<std::chrono::milliseconds>(duration).count() << " milliseconds" << endl;
   
}

void test_cudaMallocManaged(size_t N){
    char* tmp;

    CHECK_CUDA(cudaMallocManaged((void**)&tmp, N, cudaMemAttachGlobal)); 
    for(size_t i =0;i<N;i++){
        tmp[i] = 0;
    }
    
    int grid = (int)((N+255)/256);
    add1<<< grid, 256>>>(tmp, N);

    auto start = std::chrono::high_resolution_clock::now();
    
    add1_simd(tmp, N);
    auto end = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaFree(tmp));
    auto duration = end - start;
    cout << "Time taken cudaMallocManaged: " << duration_cast<std::chrono::milliseconds>(duration).count() << " milliseconds" << endl;
   
}


int main(){
    size_t N = 33554432 * 16;
    // define vector for cache-size

    test_cpu(N);
    
    test_cudaMallocHost(N);
    
    test_cudaMallocManaged(N);

    return 0;
}