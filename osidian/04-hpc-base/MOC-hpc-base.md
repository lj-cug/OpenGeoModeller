---

tags: [MOC, hpc-base模块, OpenGeoModeller]

aliases: [hpc-base知识图谱入口]

---



# MOC · hpc-base



高性能计算基座文档图谱。上级枢纽：[[hpc-base]]。总库入口：[[MOC-OpenGeoModeller]]。



## 如何看图



1. 关系图谱过滤：`tag:#hpc-base模块`

2. Canvas：[[04-hpc-base/hpc-base-知识图谱.canvas|hpc-base-知识图谱]]

3. 文字关系：[[hpc-base模块关系]]



## 按分层



### 运行环境



- [[Cluster-DIY]]



### 构建与语言



- [[Build]] · [[Language]] · [[garbage]]（非主线）



### 并行与加速



- [[CUDA]] · [[OpenCL]] · [[MPI]]



### 求解与 IO



- [[Linear_Solver]] · [[HPC-IO]]



### 高层抽象



- [[DSL]] · [[HPX]]



## 连向父仓库学科（桥接）



- [[DSL]] → [[SurfaceWater]] · [[Geophysics]] · [[ShallowWater]]

- [[Linear_Solver]] → [[Underground]] · [[Hemodynamics]] · [[SurfaceWater]]

- [[CUDA]] → [[ShallowWater]] · [[VirtualReality]] · [[Turbulence]]

- [[MPI]] / [[Build]] → 多数并行学科

- [[HPC-IO]] → [[Meteorology]] · [[VirtualReality]]



## 仓库对照



- `hpc-base/README.md`

- `docs/superpowers/specs/2026-08-14-hpc-base-markdown-design.md`

