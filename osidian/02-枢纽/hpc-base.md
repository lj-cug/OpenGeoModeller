---
tags: [枢纽, 基座, 学科, OpenGeoModeller]
aliases: [高性能计算基座, HPC]
role: 算力基座
repo_path: hpc-base/
---

# hpc-base

算力与软件栈基座：编译器、MPI、PETSc、CUDA、容器、DSL。几乎所有学科入口的「相关文档」都链到本模块。

## 内部模块（展开图谱）

→ [[MOC-hpc-base]] · [[hpc-base模块关系]] · [[04-hpc-base/hpc-base-知识图谱.canvas|hpc-base-知识图谱]]

| 分层 | 模块 |
|------|------|
| 运行环境 | [[Cluster-DIY]] |
| 构建 / 语言 | [[Build]] · [[Language]] · [[garbage]] |
| 并行加速 | [[CUDA]] · [[OpenCL]] · [[MPI]] |
| 求解 / IO | [[Linear_Solver]] · [[HPC-IO]] |
| 高层抽象 | [[DSL]] · [[HPX]] |

## 服务对象（父仓库学科）

- [[Meteorology]] [[Hydrology]] [[SurfaceWater]] [[ShallowWater]]
- [[Underground]] [[Geophysics]] [[Geological-Modelling]]
- [[ESM-Coupler]] [[Turbulence]] [[Hemodynamics]] [[Meshing]] [[VirtualReality]] [[GroundWater]]

## 典型桥接

- [[DSL]] → [[SurfaceWater]] / [[Geophysics]] / [[ShallowWater]]
- [[Linear_Solver]] → [[Underground]] / [[Hemodynamics]]
- [[CUDA]] / [[MPI]] → 多数并行模式

## 仓库文档

- `hpc-base/` · `hpc-base/README.md`
- `docs/superpowers/specs/2026-08-14-hpc-base-markdown-design.md`

## Agent 开发桥接

- [[agent-dev]] · [[Skills]] · [[主题-PETSc-AmgX-Agent]] · [[主题-Fortran-Agent]]

## 索引

- [[MOC-OpenGeoModeller]] · [[学科关系总览]]
