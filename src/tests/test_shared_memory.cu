//------------------------------------------------------------------------------------------------------------
#include <iostream>
#include <chrono>
#include <vector>
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
size_t L3CACHE = 33554432;
//------------------------------------------------------------------------------------------------------------
void warmup_cache(){
    std::vector<char> val(L3CACHE, 0);
    for(int i=0;i<L3CACHE;i++){
        val[i] += 7;
    }
}
//------------------------------------------------------------------------------------------------------------
#define CHECK_CUDA(call) do {                                         \
  cudaError_t _e = (call);                                            \
  if (_e != cudaSuccess) {                                            \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
            cudaGetErrorString(_e));                                  \
    exit(1);                                                          \
  }                                                                   \
} while (0)
//------------------------------------------------------------------------------------------------------------
extern "C" void add1_simd(char *data, size_t N);
//------------------------------------------------------------------------------------------------------------
void test_cpu(size_t N){
    void *t = mmap(NULL, N, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    char *arr = (char*)t;
    if(t == MAP_FAILED){
        printf("allocating MMAP failed");
        exit(1);
    }
    for(size_t i =0; i<N; i++){
        arr[i] = 1;
    }
    auto start = high_resolution_clock::now();

    for(int i=0;i<10;i++){
        for(size_t i =0;i<N;i++){
            arr[i]++;
        }
    }

    auto end = high_resolution_clock::now();
    
    if(munmap(t, N) == -1){
        perror("munmap");
    }
    
    duration<double> duration = end - start;
    cout << "Time taken MMAP: " << duration_cast<std::chrono::microseconds>(duration).count() << " milliseconds" << endl;
    
}
//------------------------------------------------------------------------------------------------------------
void test_cpu2(size_t N){
    char *arr;
    posix_memalign((void **) &arr, 64, N);
    if(!arr){
        printf("allocating MMAP failed");
        exit(1);
    }
    for(size_t i =0; i<N; i++){
        arr[i] = 1;
    }
    auto start = high_resolution_clock::now();

    for(int i=0;i<10;i++){
        for(size_t i =0;i<N;i++){
            arr[i]++;
        }
    }

    auto end = high_resolution_clock::now();
    
    free(arr);
    
    duration<double> duration = end - start;
    cout << "Time taken posix_memalign: " << duration_cast<std::chrono::microseconds>(duration).count() << " milliseconds" << endl;
    
}
//------------------------------------------------------------------------------------------------------------
__global__ void add1(char* arr, size_t N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N)arr[i] += 1;
}
//------------------------------------------------------------------------------------------------------------
void test_cudaMallocHost(size_t N){
    char* data;

    CHECK_CUDA(cudaMallocHost((void**)&data, N)); 
    for(size_t i =0;i<N;i++){
        data[i] = 1;
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
    for(int i=0;i<10;i++){
        for(size_t i =0;i<N;i++){
            data[i]++;
        }
    }
    auto end = high_resolution_clock::now();

    CHECK_CUDA(cudaFreeHost(data));
    
    auto duration = end - start;
    cout << "Time taken cudaMallocHost: " << duration_cast<std::chrono::microseconds>(duration).count() << " milliseconds" << endl;
   
}
//------------------------------------------------------------------------------------------------------------
void test_cudaMallocManaged(size_t N){
    char* tmp;

    CHECK_CUDA(cudaMallocManaged((void**)&tmp, N, cudaMemAttachGlobal)); 
    for(size_t i =0;i<N;i++){
        tmp[i] = 1;
    }
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;   
    loc.id   = device;  
    int grid = (int)((N+255)/256);
    
    CHECK_CUDA(cudaMemPrefetchAsync(tmp, N, loc, 0));
   
    add1<<< grid, 256>>>(tmp, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemPrefetchAsync(tmp, N, cudaCpuDeviceId, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start = high_resolution_clock::now();
    for(int i=0;i<10;i++){
        for(size_t i =0;i<N;i++){
            tmp[i]++;
        }
    }
    auto end = high_resolution_clock::now();
    int total{};
    for(size_t i=0;i<N;i++){
        total +=tmp[i];
    }
    CHECK_CUDA(cudaFree(tmp));
    auto duration = end - start;
    cout << "Time taken cudaMallocManaged: " << duration_cast<std::chrono::microseconds>(duration).count() << " microseconds" << endl;
   
}
//------------------------------------------------------------------------------------------------------------
int main(){
    size_t L3 = 33554432;
    size_t L2 = 1048576 * 8;
    size_t L1 = 356352;
    std::vector<size_t> sizes = {L1, L2, L3};

    for(auto N : sizes){
        cout << "TEST CASES FOR SIZE[" << N <<"]:"<< endl;    
        for(int i=0;i<10;i++)warmup_cache();
        test_cpu(N);
        for(int i=0;i<10;i++)warmup_cache();
        test_cudaMallocHost(N);
        for(int i=0;i<10;i++)warmup_cache();
        test_cudaMallocManaged(N);
        cout << endl;
    }
    
    return 0;
}
//------------------------------------------------------------------------------------------------------------