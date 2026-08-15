---
tags: [说明, hpc-base模块, OpenGeoModeller]
aliases: [hpc-base关系图, HPC模块关系]
---

# hpc-base 模块关系

依据 `hpc-base/README.md` 与 `docs/superpowers/specs/2026-08-14-hpc-base-markdown-design.md`，以及各一级模块 README「相关文档」。

## 逻辑分层

```mermaid
flowchart TB
  cluster[Cluster-DIY]
  build[Build]
  lang[Language]
  garbage[garbage 非主线]
  parallel[CUDA / OpenCL / MPI]
  solver[Linear_Solver]
  io[HPC-IO]
  dsl[DSL / HPX]

  cluster --> build
  cluster --> lang
  cluster --> garbage
  build --> parallel
  lang --> parallel
  parallel --> solver
  parallel --> io
  solver --> dsl
  io --> dsl
```

## 依赖速查

| 模块 | 应链接 |
|------|--------|
| [[Cluster-DIY]] | [[MPI]] · [[Build]] · [[garbage]] · [[CUDA]] |
| [[Build]] | [[Language]] · [[CUDA]] · [[Cluster-DIY]] |
| [[Language]] | [[Build]] · [[MPI]] · [[DSL]] |
| [[CUDA]] | [[MPI]] · [[HPC-IO]] · [[Linear_Solver]] · [[OpenCL]] |
| [[OpenCL]] | [[CUDA]] · [[Linear_Solver]] |
| [[MPI]] | [[CUDA]] · [[Cluster-DIY]] · [[Linear_Solver]] · [[Language]] |
| [[Linear_Solver]] | [[CUDA]] · [[OpenCL]] · [[MPI]] · [[Build]] |
| [[HPC-IO]] | [[CUDA]] · [[MPI]] · [[Build]] |
| [[DSL]] | [[Language]] · [[Build]] · [[Linear_Solver]]；父仓 [[SurfaceWater]] / [[Geophysics]] |
| [[HPX]] | [[Build]] · [[Language]] · [[MPI]] |
| [[garbage]] | [[Cluster-DIY]] · [[Build]]（归档） |

## 关键交叉点

- **MPI-GPUDirect** ← 依赖 → **CUDA/GPUDirect**
- **AmgX** ↔ CUDA；**AmgCL** ↔ OpenCL；**PETSc** ↔ MPI
- **HPC-IO / GDS** ↔ GPUDirect Storage
- **DSL/OP2** ↔ 父仓 [[ShallowWater]]；**Firedrake/Devito** ↔ [[SurfaceWater]] / [[Geophysics]]

## 可视化

- Canvas：[[04-hpc-base/hpc-base-知识图谱.canvas|hpc-base-知识图谱]]
- 总入口：[[MOC-hpc-base]] · [[MOC-OpenGeoModeller]]
