# SCHISM

## 介绍

1. 美国弗吉尼亚海洋研究所开发的三维非结构网格模式的海洋动力学模式，有限单元法

2. 编程语言：FORTRAN, C语言

3. 并行方式：集群并行MPI和多线程并行OpenMP

## 内容

- [manual](./manual/)：模式的前处理程序和后处理，以及测试算例等
- [install](./install/)：源码编译安装的脚本程序和源码压缩包
- [doc](./doc/)：模式的基本介绍、编译步骤和算例测试说明等
- [pre-processing](./pre-processing/)：前处理
- [post-processing](./post-processing/)：后处理
- [SCHISM-5.9.0-PDAF](./SCHISM-5.9.0-PDAF/)：v5.9.0 与 PDAF 相关材料

注意：早期使用了版本 v5.3.1 ~ v5.6.1; v5.9.0以后版本的I/O有很大的改进

## 相关文档

- [SurfaceWater](../)：学科入口
- [OCSMesh](../../Meshing/OCSMesh/)：SCHISM 非结构网格生成
- [install/build-SCHISM_v5.9.0.md](./install/build-SCHISM_v5.9.0.md)：v5.9.0 编译
- [hpc-base](../../hpc-base/)：编译器、MPI、PETSc、容器等
