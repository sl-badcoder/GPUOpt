# 🚀 Optimizing CUDA Memory Layouts
---

## Which Memory Layouts exist in CUDA?
In Cuda there exists several possibilities to allocate shared memory, we will mainly look at the following three:

 `Memory Layout` | `Prefetching Possible?` |`Pinning Possible?` | `Oversubscribing VRAM?` 
 :--- | :---: | :---: | ---: 
 `Pinned Memory` | :x: | :white_check_mark: | :x:
 `Mapped Memory` | :x: | :x: | :white_check_mark: 
 `Unified Memory` | :white_check_mark: | :white_check_mark: | :white_check_mark: 

## 🔎 How should we work with Unified Memory?
When working in the heterogeneous world where we can access the GPU and CPU simoultanesly the question arise on how we should allocate our shared memory. There are several ways for allocating this memory. In this repo we will mainly look at Unified Memory and Mapped Memory, since both are options to oversubscribe our VRAM without complex code changes. 

## 🔎 Why should we look at oversubscribing VRAM?
For memory intensive workloads like sorting or training DNN VRAM gets heavily pressured. Especially for 3D generation neural networks (e.g. MVDream_threestudio) eat up like 20GiB of VRAM which isn't feasable on a normal RTX 5090. Upgrading to a larger GPU results in additional cost expenses. Exploitiong physical RAM would in this case reduce costs. 
