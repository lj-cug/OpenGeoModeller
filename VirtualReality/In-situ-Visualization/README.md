# 原位可视化
Catalyst的示例代码及原理讲义课件

# 应用程序
`
RegESM：飓风的移动过程，基于paraview-catalyst
BloodFlow：分子动力学LAMMPS + LBM模拟后端Palabos库 + SENSEI在线可视化系统，实现血管中红细胞运动的模拟及可视化
Nek5000 (2021, KTH Sweden)
CESM_1.0
miniFE
`

# 参考文献
libMesh-sedimentation： Jose J. Camata, et al. In situ visualization and data analysis for turbidity currents simulation. Computers and Geosciences 110 (2018) 23–31

# 总结
大规模科学计算的IO成为瓶颈的情况下，在线可视化成为当前的研究热点。
不同的硬件架构的并行计算，对在线可视化也提出了新的要求。
ParaView Catalyst和VisIT Libsim是在线可视化的先驱，已有很多CFD应用使用了Catalyst，如Nek500, RegESM, PyFR, CAM, ...
传统的在线可视化都是将数据从CPU（内存）转移到GPU内存，如Catalyst
越来越多的代码迁移到GPU（CUDA），因此如何实现在GPU中实现计算与可视化的衔接，是很具有挑战和研究意义的。
VTK-m成为衔接GPU计算与可视化的中间层软件，开展较早的研究的是PyFR-Catalyst

现有了更统一的在线可视化框架，例如SENSEI, Damaris和Ascent，
SENSEI和Ascent是美国的DOE与LLNL开发的在线可视化统一框架，Damaris是法国inria开发的在线可视化统一框架。都是通过XML配置文件和调用API来实现MPI+多核/众核的数据可视化。

## Catalyst and Libsim
ParaView和VISIT软件的原生原位可视化库

## Ascent (推荐)
Ascent研发的程序，可支持MPI+OpenMP/CUDA的在线可视化，通过VTK-m中间层。
封装了Catalyst and Libsim
Ascent支持C++和Python语言
有很多Mini-app: LULESH, Coverleaf3D等

## SENSEI
SENSEI整合了Catalyst, Ascent和ADIOS2(自适应数据管理框架)

## Damaris
Damaris是一个轻量级的在线可视化统一框架，可以对FORTRAN/C/C++的模拟代码，快速实施VisIT与ParaView的在线可视化。
Damaris-1.5.0开始支持ParaView下的非结构化网格数据的在线可视化，没有1.4版本发布。暂不支持VisIT的非结构网格数据的可视化。
没有公开的tutorials !

## ADIOS2
In transit analysis:  ADIOS2-Catalyst
