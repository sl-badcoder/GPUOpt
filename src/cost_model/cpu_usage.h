#ifndef CPU_USAGE_H
#define CPU_USAGE_H
//------------------------------------------------------------------------------------------------------------
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
//------------------------------------------------------------------------------------------------------------
using ll = long long;
const int line_size = 10;/// current linux version >= 2.6.33
//------------------------------------------------------------------------------------------------------------
inline int getNumberCPUs(){
    // make more reliable
    std::ifstream file("/proc/stat");
    std::string start;
    file >> start;
    int NUM_CPUS =0;
    std::vector<ll> vals(line_size, 0);
    for(auto& val : vals){
        file >> val;
    }
    file >> start;
    while(start.substr(0,3) == "cpu"){
        NUM_CPUS++;
        for(auto& val : vals){
            file >> val;
        }
        file >> start;
    }
    return NUM_CPUS;
}
//------------------------------------------------------------------------------------------------------------
inline double getCurrentCPUidle(){
    std::ifstream file("/proc/stat");

    // vals[3] == idle time
    std::string name;
    std::vector<ll> vals(line_size, 0);
    file >> name;
    
    for(auto& val : vals){
        file >> val;
    }
    
    ll sum_all = 0;
    
    for(int i=0;i<vals.size();i++){
        sum_all += vals[i];
    }
    //cout << name;
    return double(vals[3] * 100) / double(sum_all);
}
//------------------------------------------------------------------------------------------------------------
inline double getCurrentCPUusage(){
    return 100 - getCurrentCPUidle();
}

inline int getFreeRAM(){
    std::ifstream mem_file("/proc/meminfo");
    std::string mem_t_s, mem_f_s, t;
    ll mem_t, mem_f;
    mem_file >> mem_t_s >> mem_t >> t;
    mem_file >> mem_t_s >> mem_t >> t;

    mem_file >> mem_f_s >> mem_f >> t;
    return mem_f / (1024.0);
}
//------------------------------------------------------------------------------------------------------------
// think about how to make code as malleable as possible
//------------------------------------------------------------------------------------------------------------
#endif