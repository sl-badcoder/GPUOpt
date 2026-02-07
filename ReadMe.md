# Bitonic Sort in Heterogeneous Systems  
### Exploring Hybrid CPU/GPU Execution and Unified Memory

## Overview
This repository contains a research project on **Bitonic Sort** in **heterogeneous software architectures**, focusing on **CPU–GPU co-execution** and **memory management strategies**.  
While Bitonic Sort has a higher asymptotic complexity than comparison-based or radix-based sorting algorithms, its **regular, data-independent structure** makes it highly suitable for parallel execution on modern SIMD CPUs and GPUs.

The core goal of this project is to investigate how Bitonic Sort can be **optimized, hybridized, and used as an evaluation vehicle** to study:
- CPU vs. GPU execution trade-offs
- Unified Memory (UM) vs. explicit data movement
- Chunked processing vs. oversubscription
- Memory overlap, page migration, and prefetching strategies

---

## Motivation
Modern systems increasingly rely on **heterogeneous hardware**, where CPUs and GPUs coexist with different performance characteristics:
- CPUs excel at **low-latency**, branch-heavy workloads and small working sets
- GPUs excel at **high-throughput**, massively parallel workloads

Bitonic Sort is particularly interesting in this context because:
- Its execution pattern evolves from **small, local merges** to **large, global merges**
- Early stages can benefit from CPU execution
- Later stages benefit from GPU throughput
- Its access pattern stresses **memory systems**, making it ideal for studying **Unified Memory behavior**
<p align="center">
  <img src="images/bitonic_sort_architecture.png" width="600">
</p>
---

## Research Focus
This project investigates the following dimensions:

### 1. Hybrid CPU/GPU Bitonic Sort
When we exceed VRAM size we pass over the array (VRAM_SIZE) and do the later large pass over the whole array on the CPU
 **dynamic hybrid model**:
- CPU executes pre-processing and large merging step at the end
- GPU executes most of the work as long as it fits in VRAM


### 2. Unified Memory vs. Explicit Data Movement
A major part of the research evaluates:
- CUDA Unified Memory with and without `cudaMemPrefetchAsync`
- Memory advice (`cudaMemAdvise` - 3 types)
- Pinned (page-locked) memory
- Explicit chunk-based processing (is it better to split up the arry beforehand)


### 3. Memory Access Patterns & Oversubscription
We look at different access patterns like having linear, wide-range access or wide/small-range access(Bitonic-Sort):
- Which optimization method is for each access pattern the best?
- How much data should we prefetch?
- Can memAdvise help improve performance especially when oversubscribnig


---

## Current Status
- Baseline CPU and GPU Bitonic Sort implementations
- Shared-memory GPU optimizations (tile-based execution)
- Hybrid CPU/GPU bitonic Sort
- Extensive profiling of:
  - Unified Memory
  - Pinned memory
  - Oversubscription scenarios
  - Prefetching and memory advice strategies
  - access patterns like linear/wide-range


---
