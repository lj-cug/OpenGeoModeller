# ROMS

ROMS是曲线结构网格的3D海洋动力学模式，在物理海洋学领域应用广泛的模式

FORTRAN语言

ROMS与很多数值气象模式实现了耦合

## agrif

AMR库，应用于ROMS，形成ROMS-Agrif，现在演进为CROCO模式

## 本仓库文档

- [install](./install/)：编译与运行脚本
- [doc](./doc/)：版本介绍、输入文件与工具说明
- [pyroms](./pyroms/)：Python 前后处理

## 相关文档

- [SurfaceWater](../)：学科入口
- [CROCO](../CROCO/)：由 ROMS-Agrif 演进而来
- [Meteorology/WRFV4](../../Meteorology/WRFV4/)：常与 ROMS 耦合的大气模式
- [ESM-Coupler](../../ESM-Coupler/)：耦合框架
- [hpc-base](../../hpc-base/)：编译器、MPI 等
