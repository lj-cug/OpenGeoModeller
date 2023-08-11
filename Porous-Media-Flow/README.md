# Porous-Media-Flow

多孔介质流体力学模型，包括地下水和油藏储层模拟等，例如：MODFLOW, OPM, ...

## OPM

基于DUNE库、Zoltan库开发的黑油模型(Black Oil model)

## DuMux

基于DUNE库的多孔介质流体模拟程序

## GEOSX

美国Lawrence Livermore National Laboratory (LLNL), Stanford University, TotalEnergies, Chevron 开发的新一代地质碳封存及其他地下能源系统的模拟程序

## OpenGeoSys

Thermo-hydro-mechanical/chemical (THMC)模拟

## MODFLOW6

美国USGS开发的地下水模拟系统，基于PETSc，MPI并行化MODFLOW6

## MPLBM-UT

以格子Boltzmann方法的palabo库为求解后端，使用CT扫描的岩石样品的图像数据为输入，模拟孔隙流动下的毛细压力与相对渗透率等的物理过程

## LBDEMcoupling

格子Boltzmann方法求解库Palabos与离散元模型LIGGGHTS耦合，可模拟床面泥沙颗粒的起动，以及水沙两相流的介观模型

# 地质建模

使用近年来开源的隐式地质建模Python脚本工具，实现地质建模（即多孔介质流动模型的前处理）

## Implicit-GeoModeling (隐式地质建模与显式地质建模)

目前多数的地质建模软件，如Petrel和SKUA-GOCAD，都是使用显式地质建模，也就是建模过程必须显式地构建出地质体，如断层，需要大量的手工工作量

而隐式地质建模，仅需要地质体的梯度或dip-azimuth等地震属性数据的势场插值，就能得到地质建模文件，无需手工介入，但由于对计算机内存和计算性能要求较高，近5年来才有可用的计算机程序供使用

当前的隐式建模Python工具主要是：GemPy和LoopStructural

## Gempy

德国亚琛工业大学开发的隐式地质建模Python程序

## LoopStructural

澳大利亚开发的隐式地质建模Python程序

## CPM

角点网格的数据结构及格式转换
