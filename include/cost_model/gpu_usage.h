#ifndef GPU_USAGE_H
#define GPU_USAGE_H
//------------------------------------------------------------------------------------------------------------
#include <cuda_runtime.h>
//------------------------------------------------------------------------------------------------------------
// code: https://forums.developer.nvidia.com/t/cudamemgetinfo-how-does-it-work/21921/2
//------------------------------------------------------------------------------------------------------------
inline size_t getGPUFreeMemory(){
    size_t free_m;
    size_t free_t,total_t;

    cudaMemGetInfo(&free_t,&total_t);
    //free_m = free_t/(size_t)(1048576.0);

    return free_t;
}
//------------------------------------------------------------------------------------------------------------
inline double getGPUMemoryUsagePercent(){
    float free_m, total_m, used_m;
    size_t free_t,total_t;
    cudaMemGetInfo(&free_t,&total_t);

    free_m = free_t/(8*1048576.0);
    total_m=total_t/(1048576.0);
    
    used_m = total_m - free_m;
    return double(used_m) / double(total_m);
}
//------------------------------------------------------------------------------------------------------------
inline double getGPUFreeMemoryPercent(){
    float free_m, total_m;
    size_t free_t,total_t;
    cudaMemGetInfo(&free_t,&total_t);

    free_m = free_t/1048576.0;
    total_m=total_t/1048576.0;
    
    return double(free_m) / double(total_m);
}
//------------------------------------------------------------------------------------------------------------
#endif
//------------------------------------------------------------------------------------------------------------