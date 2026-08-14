# Meteorology

数值天气预报 (NWP) 及相关工具。

## 子目录

- [WRFV4](./WRFV4/)：数值天气预报模型，持续开发中（4.x）
- [RegCM4](./RegCM4/)：区域性 NWP；较新版本增加非静水压力模块
- [MPAS_v7.0](./MPAS_v7.0/)：非结构网格新一代 NWP，理论与 WRF 类似
- [HWRF4](./HWRF4/)：NCEP 飓风跟踪模式，基于 WRF 框架
- [NCL_v6.4](./NCL_v6.4/)：NCAR NCL 脚本，用于气象结果可视化
- [GoogleEarthEngine](./GoogleEarthEngine/)：GEE 安装与使用笔记

## 相关文档

- [ESM-Coupler](../ESM-Coupler/)：与海洋 / 波浪等耦合（COAWST、RegESM、CoastalApp）
- [Hydrology](../Hydrology/)：WRF-Hydro 等陆面 / 水文耦合
- [SurfaceWater](../SurfaceWater/)：风暴潮 / 波浪下游应用
- [hpc-base](../hpc-base/)：编译、MPI、NetCDF 等
