//------------------------------------------------------------------------------------------------------------
#include "cpu_usage.h"
#include "gpu_usage.h"
//------------------------------------------------------------------------------------------------------------
using std::cout;
using std::endl;
using std::cin;
//------------------------------------------------------------------------------------------------------------
// simple dummy cost model
// build up on this code to make more complex decisions
// assumptions
//------------------------------------------------------------------------------------------------------------
int main(){

    // idea: run as long as possible on GPU if memory not available run on CPU
    cout << "CPU Free memory: " << getFreeRAM() << " MB" << endl;
    cout << "GPU Free memory: " << getGPUFreeMemory() << " MB" << endl;
    cout << "CPU idle time: " << getCurrentCPUidle() << endl;
    cout << "NUM_CPUS: " << getNumberCPUs() << endl;

    return 0;
} 
//------------------------------------------------------------------------------------------------------------