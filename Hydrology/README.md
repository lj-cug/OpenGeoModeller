# Hydrology

分布式水文模型及其耦合模型。

## 子目录

- [PIHM](./PIHM/)：OpenMP 并行的三角形非结构网格分布式水文模型（PIHM2.x、PIHM4.0 等）
- [SHUD](./SHUD/)：与 PIHM 相关的非结构网格水文模型
- [ParFLOW](./ParFLOW/)：地下水模型（C 核心，TCL / Python 脚本）
- [GSFLOW](./GSFLOW/)：USGS 地表水与 MODFLOW 的耦合模型（pywatershed 的前身）
- [CHM](./CHM/)：加拿大开发的非结构网格冰雪下垫面分布式水文模型
- [pywatershed](./pywatershed/)：USGS 耦合 PRMS 与 MODFLOW6 的 Python 工作流
- [UniFHy](./UniFHy/)：英国开发的地表水–地下水耦合水文模型（Python）
- [tRIBS](./tRIBS/)：基于 TIN 的分布式水文模型（C++，MPI）
- [Raven](./Raven/)：Waterloo 大学集中式 / 半分布式水文模型
- [SEIMS](./SEIMS/)：中科院地理所分布式水文模型（C++，MPI + OpenMP）
- [WRF-Hydro](./WRF-Hydro/)：与大气耦合的水文组件

## 总结

1. Python 开发的水文模型（pywatershed、UniFHy）降低了使用难度，但并行化仍是分布式水文模型的重要方向（如 ParFLOW）。
2. 非结构化网格能体现流域空间各向异性，但数据结构更复杂；目前较少用于分布式水文，主要有：PIHM、SHUD、CHM、tRIBS。
3. 并行化程度整体不高；并行模型包括：CHM、PIHM、SHUD（OpenMP）和 tRIBS、SEIMS、ParFLOW（MPI）；ParFLOW 并行手段最多（MPI + OpenMP / CUDA）。

## 相关文档

- [Underground](../Underground/)：MODFLOW6 等地下水 / 多孔介质模型
- [Meshing](../Meshing/)：三角网格 / mesher 等前处理
- [Meteorology](../Meteorology/)：大气强迫与 WRF-Hydro 上游
- [hpc-base](../hpc-base/)：MPI、OpenMP、CUDA 等
