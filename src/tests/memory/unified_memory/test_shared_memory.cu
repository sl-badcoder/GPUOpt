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
void arrayTest(char* tmp, size_t N){
    //for(int i=0;i<10;i++){
        for(size_t i =0;i<N;i+=100){
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
__global__ void add(char* arr, size_t start, size_t end){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<end && i >= start)arr[i] += 1;
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
    //CHECK_CUDA(cudaMemAdvise((void*)tmp, N, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));

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

    //CHECK_CUDA(cudaMemPrefetchAsync(tmp, N, cudaCpuDeviceId, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start = high_resolution_clock::now();


    //for(int i=0;i<10;i++){
        for(size_t i =0;i<N;i++){
            tmp[i]+=1;
        }
    //}

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
// TEST RAM->VRAM->RAM->VRAM copy pattern
//------------------------------------------------------------------------------------------------------------
void test_ramVramPattern(size_t N){

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaStream_t s2;
    cudaStreamCreate(&s2);
    char* tmp;
    auto start = high_resolution_clock::now();
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
    CHECK_CUDA(cudaMemAdvise((void*)tmp, N, cudaMemAdviseSetAccessedBy, loc.id));
    CHECK_CUDA(cudaMemAdvise((void*)tmp, N, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));

    // RAM -> VRAM
    //CHECK_CUDA(cudaMemPrefetchAsync(tmp, N, device, s2));
    //cudaStreamSynchronize(stream);
    //CHECK_CUDA(cudaDeviceSynchronize());    
    add1<<< grid, 256, 0, stream>>>(tmp, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    // VRAM -> RAM
    //CHECK_CUDA(cudaMemPrefetchAsync(tmp, N, cudaCpuDeviceId, s2));
    //cudaStreamSynchronize(stream);
    arrayTest(tmp, N);
    // RAM -> VRAM
    //CHECK_CUDA(cudaMemPrefetchAsync(tmp, N, device, s2));
    //cudaStreamSynchronize(stream);
    add1<<< grid, 256, 0, stream>>>(tmp, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    // VRAM -> RAM
    //CHECK_CUDA(cudaMemPrefetchAsync(tmp, N, cudaCpuDeviceId, s2));
    //cudaStreamSynchronize(stream);
    arrayTest(tmp, N);


    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaStreamSynchronize(s2);
    cudaStreamDestroy(s2);
    auto end = high_resolution_clock::now();
    CHECK_CUDA(cudaFree(tmp));
    auto duration = end - start;
    cout << "Time taken RAM-VRAM-PATTERN: " << duration_cast<std::chrono::microseconds>(duration).count() << " seconds" << endl;
}
void testRAMVRAM(size_t N){
    cudaStream_t s1;
    cudaStream_t s2;
    cudaStream_t s3;

    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    cudaStreamCreate(&s3);

    char* tmp;
    auto start = high_resolution_clock::now();
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

    for(int i=0;i<10;i++){
        size_t st = i * N/10;
        add<<< grid, 256, 0, s1>>>(tmp, st, st + N / 10);        
    }

    for(int i=0;i<10;i++){
        size_t st = i * N / 10;
        //cudaStreamSynchronize(s2);
        CHECK_CUDA(cudaMemPrefetchAsync(tmp + st, N / 10, device, s2));
    }
    //cudaStreamSynchronize(s2);
    for(int i=0;i<10;i++){
        //cudaStreamSynchronize(s3);
        size_t st = i * N / 10;
        CHECK_CUDA(cudaMemPrefetchAsync(tmp + st, N / 10, cudaCpuDeviceId, s3));
    }
    //cudaStreamSynchronize(s3);

    arrayTest(tmp, N);
    CHECK_CUDA(cudaStreamDestroy(s1));
    CHECK_CUDA(cudaStreamDestroy(s2));
    CHECK_CUDA(cudaStreamDestroy(s3));
    cudaDeviceSynchronize();
    auto end = high_resolution_clock::now();
    CHECK_CUDA(cudaFree(tmp));
    auto duration = end - start;
    cout << "Time taken RAM-VRAM: " << duration_cast<std::chrono::microseconds>(duration).count() << " seconds" << endl;

}
//------------------------------------------------------------------------------------------------------------
void test_ramVramPatternPinned(size_t N){

    char* data;
    auto start = high_resolution_clock::now();
    // RAM -> VRAM
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
    // VRAM -> RAM
    CHECK_CUDA(cudaFreeHost(data));
    char* data2;
    CHECK_CUDA(cudaMallocHost((void**)&data2, N)); 
    CHECK_CUDA(cudaMemcpyAsync(data2, dbuf, (size_t)N , cudaMemcpyDeviceToHost, s));
    CHECK_CUDA(cudaStreamSynchronize(s));
    CHECK_CUDA(cudaFree(dbuf));
    CHECK_CUDA(cudaStreamDestroy(s));
    arrayTest(data2, N);
    // RAM -> VRAM
    char* dbuf2 = NULL;
    CHECK_CUDA(cudaMalloc((void**)&dbuf2, (size_t)N));
    cudaStream_t s2;
    CHECK_CUDA(cudaStreamCreate(&s2));
    CHECK_CUDA(cudaMemcpyAsync(dbuf2, data2, (size_t)N , cudaMemcpyHostToDevice, s2));
    add1<<< grid, 256, 0, s2>>>(dbuf2, N);
    CHECK_CUDA(cudaGetLastError());
    // VRAM -> RAM
    CHECK_CUDA(cudaFreeHost(data2));
    char* data3;
    CHECK_CUDA(cudaMallocHost((void**)&data3, N)); 
    CHECK_CUDA(cudaMemcpyAsync(data3, dbuf2, (size_t)N , cudaMemcpyDeviceToHost, s2));
    CHECK_CUDA(cudaStreamSynchronize(s2));
    CHECK_CUDA(cudaFree(dbuf2));
    CHECK_CUDA(cudaStreamDestroy(s2));
    arrayTest(data3, N);   

    
    auto end = high_resolution_clock::now();



    CHECK_CUDA(cudaFreeHost(data3));
    
    auto duration = end - start;
    cout << "Time taken patternPinned: " << duration_cast<std::chrono::microseconds>(duration).count() << "  microseconds" << endl;
}
//------------------------------------------------------------------------------------------------------------
void test_ramVramPatternPinned2(size_t N){

    char* data;
    auto start = high_resolution_clock::now();
    // RAM -> VRAM
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
    // VRAM -> RAM

    CHECK_CUDA(cudaMemcpyAsync(data, dbuf, (size_t)N , cudaMemcpyDeviceToHost, s));
    CHECK_CUDA(cudaStreamSynchronize(s));
    CHECK_CUDA(cudaFree(dbuf));
    CHECK_CUDA(cudaStreamDestroy(s));
    arrayTest(data, N);
    // RAM -> VRAM
    char* dbuf2 = NULL;
    CHECK_CUDA(cudaMalloc((void**)&dbuf2, (size_t)N));
    cudaStream_t s2;
    CHECK_CUDA(cudaStreamCreate(&s2));
    CHECK_CUDA(cudaMemcpyAsync(dbuf2, data, (size_t)N , cudaMemcpyHostToDevice, s2));
    add1<<< grid, 256, 0, s2>>>(dbuf2, N);
    CHECK_CUDA(cudaGetLastError());
    // VRAM -> RAM
    CHECK_CUDA(cudaMemcpyAsync(data, dbuf2, (size_t)N , cudaMemcpyDeviceToHost, s2));
    CHECK_CUDA(cudaStreamSynchronize(s2));
    CHECK_CUDA(cudaFree(dbuf2));
    CHECK_CUDA(cudaStreamDestroy(s2));
    arrayTest(data, N);   

    
    auto end = high_resolution_clock::now();
    CHECK_CUDA(cudaFreeHost(data));

    auto duration = end - start;
    cout << "Time taken patternPinned: " << duration_cast<std::chrono::microseconds>(duration).count() << "  microseconds" << endl;
}
void test_ramVramPatternPinned2_(size_t N){
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    char* data;
    auto start = high_resolution_clock::now();
    // RAM -> VRAM
    CHECK_CUDA(cudaMallocManaged((void**)&data, N , cudaMemAttachGlobal)); 
    /**for(size_t i =0;i<N;i++){
        data[i] = 1;
    }*/
    int grid = (int)((N+255)/256);
    //CHECK_CUDA(cudaMemPrefetchAsync(data, N, dev));
    add1<<< grid, 256>>>(data, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    // VRAM -> RAM
    //CHECK_CUDA(cudaMemPrefetchAsync(data, N, cudaCpuDeviceId));
    CHECK_CUDA(cudaDeviceSynchronize());
    arrayTest(data, N);
    // RAM -> VRAM
    //CHECK_CUDA(cudaMemPrefetchAsync(data, N, dev));
    add1<<< grid, 256>>>(data, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    // VRAM -> RAM
    //CHECK_CUDA(cudaMemPrefetchAsync(data, N, cudaCpuDeviceId));
    CHECK_CUDA(cudaDeviceSynchronize());
    arrayTest(data, N);   

    CHECK_CUDA(cudaFree(data));

    auto end = high_resolution_clock::now();
    
    auto duration = end - start;
    cout << "Time taken pattern unified: " << duration_cast<std::chrono::microseconds>(duration).count() << "  microseconds" << endl;
}
//------------------------------------------------------------------------------------------------------------
void test_ramVramPatternPinned3(size_t N){

    char* data;
    auto start = high_resolution_clock::now();
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    // RAM -> VRAM
    CHECK_CUDA(cudaMallocManaged((void**)&data, N , cudaMemAttachGlobal)); 
    /**for(size_t i =0;i<N;i++){
        data[i] = 1;
    }*/

    int grid = (int)((N+255)/256);
    CHECK_CUDA(cudaMemPrefetchAsync(data, N, dev));
    add1<<< grid, 256>>>(data, N);
    CHECK_CUDA(cudaGetLastError());
    //CHECK_CUDA(cudaDeviceSynchronize());
    // VRAM -> RAM
    CHECK_CUDA(cudaFree(data));
    CHECK_CUDA(cudaMallocManaged((void**)&data, N , cudaMemAttachGlobal)); 
    /**for(size_t i =0;i<N;i++){
        data[i] = 1;
    }*/
    CHECK_CUDA(cudaMemPrefetchAsync(data, N, cudaCpuDeviceId));
    CHECK_CUDA(cudaDeviceSynchronize());
    arrayTest(data, N);
    // RAM -> VRAM
    CHECK_CUDA(cudaMemPrefetchAsync(data, N, dev));
    add1<<< grid, 256>>>(data, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    //CHECK_CUDA(cudaDeviceSynchronize());
    // VRAM -> RAM
    CHECK_CUDA(cudaFree(data));
    CHECK_CUDA(cudaMallocManaged((void**)&data, N , cudaMemAttachGlobal)); 
    /**for(size_t i =0;i<N;i++){
        data[i] = 1;
    }*/
    CHECK_CUDA(cudaMemPrefetchAsync(data, N, cudaCpuDeviceId));
    CHECK_CUDA(cudaDeviceSynchronize());
    arrayTest(data, N);   

    
    CHECK_CUDA(cudaFree(data));
    auto end = high_resolution_clock::now();

    auto duration = end - start;
    cout << "Time taken pattern unified: " << duration_cast<std::chrono::microseconds>(duration).count() << "  microseconds" << endl;
    
}

//------------------------------------------------------------------------------------------------------------
int main(){
    size_t L3 = 33554432;
    size_t L2 = 1048576 * 8;
    size_t L1 = 356352;

    size_t GiB = 1073741824;
    std::vector<size_t> sizes = {L1, L2, L3, 4 * GiB, 8 * GiB, 16 * GiB};
    /**for(auto sz : sizes){
        std::cout << "SIZES: " << sz << std::endl;
        warmup_cache();
        test_ramVramPatternPinned2_( sz);
        warmup_cache();
        test_ramVramPatternPinned3( sz);
    }
    warmup_cache();
    test_cudaMallocManaged(L1);
**/
    test_ramVramPattern(268435456 * 4);

    //testRAMVRAM(8 * GiB);
    /**for(auto N : sizes){
        cout << "TEST CASES FOR SIZE[" << N <<"]:"<< endl;    
        for(int i=0;i<10;i++)warmup_cache();
        test_cpu(N);
        for(int i=0;i<10;i++)warmup_cache();
        test_cudaMallocHost(N);
        for(int i=0;i<10;i++)warmup_cache();
        test_cudaMallocManaged(N);
        cout << endl;
    }**/
    //test_ramVramPattern(0.01 * GiB);
    //warmup_cache();

    //test_ramVramPatternPinned2(8 * GiB);
    return 0;
}
//------------------------------------------------------------------------------------------------------------