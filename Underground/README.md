# Porous-Media-Flow

已经有很多开源的Darcy尺度的多孔介质流动和输移过程模拟程序，见文献（Bilke,
2019）的罗列。多孔介质流体力学模型，包括地下水和油藏储层模拟等，例如：MODFLOW, OPM, ...

## 参考文献

L. Bilke, B. Flemisch, T. Kalbacher, O. Kolditz, R. Helmig, T. Nagel, Development of open-source porous media simulators: Principles and experiences, Transp. Porous Media (2019)

## porousMultiphaseFoam

在OpenFOAM-v2206框架的基础上，开发的平面2D/3D Richards方程求解器

## MODFLOW6

美国USGS开发的地下水模拟系统，基于PETSc和MPI并行化MODFLOW6

## OPM

基于DUNE、Zoltan库开发的黑油模型(Black Oil model)，有开源的后处理程序ReInsight。用于替代商业软件ECLIPSE

## DuMux

基于DUNE库的多孔介质流体模拟程序，是OPM模拟系统的基础

## GEOSX

美国Lawrence Livermore National Laboratory (LLNL), Stanford University, TotalEnergies, Chevron 开发的新一代地质碳封存及其他地下能源系统的模拟程序

## OpenGeoSys

Thermo-hydro-mechanical/chemical (THMC)模拟系统，有ogs5和ogs6两个版本

## Waiwera

基于MPI和PETSc开发的地热及CO2封存模拟系统

## golem

基于MOOSE框架的并行化地热模拟系统

## MPLBM-UT

以格子Boltzmann方法的palabo库为求解后端，使用CT扫描的岩石样品的图像数据为输入，模拟孔隙流动下的毛细压力与相对渗透率等的物理过程

## LBDEMcoupling

格子Boltzmann方法求解库Palabos与离散元模型LIGGGHTS耦合，可模拟床面泥沙颗粒的起动，以及水沙两相流的介观模型