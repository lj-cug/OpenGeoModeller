# AMGCL介绍

## 简介

AMGCL库，是由俄国人Denis
Demidov开发的，仅使用头文件和代数多重网格算法，求解大型稀疏线性方程组的C++库。AMGCL库可作为黑箱子求解器，解决各种计算问题，因为不需要知道潜在几何的任何信息。

AMGCL提供共享式内存和分布式内存并行机制。使用内建的数据结构构建多重网格层级，然后传递给提供的后端程序。这允许使用OpenMP,
OpenCL或CUDA技术，加速求解。AMGCL库还可以提供他们自己的后端，实现AMGCL与用户代码之间的紧密整合。

尽管AMGCL库的初衷是代数多重网格，但其模块化结构，允许提供更专门的预处理器，比如CPR或Schur压力校正。可以单层预处理器用作多重网格的松弛模块。

## AMGCL设计准则

AmgCL背后的设计准则是：可用性、有效性和可扩展性。

基于策略的设计类，诸如：amgcl::make_solver或amgcl::amg，允许用户使用自己版本的迭代求解器和预处理器，很容易扩展和二次开发库（提供自己的算法）。

后端：OpenMP, OpenCL, CUDA

## 影响力

AMGCL库已广泛应用于CFD应用，如储层模拟MRST，地下水模拟等。

## 参考文献

Denis Demidov. 2020. AMGCL \-- A C++ library for efficient solution of
large sparse linear systems. Software Impacts, 6: 100037.
