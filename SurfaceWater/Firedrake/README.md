# firedrake
Python语言开发的DSL库, 基于firedrake库开发了很多地球科学领域的CFD应用程序

## 本仓库文档

- [install](./install/)：Firedrake / Thetis 安装说明
- [doc](./doc/)：软件栈等说明

## Thetis
近海岸环境的水流及泥沙输移模拟
快速实现AMR+DG+非结构网格的水流及河床演变模拟

安装见 [quick-install-thetis.md](./install/quick-install-thetis.md)

## slate
解决（隐式?)DG法的椭圆型方程组的求解问题(鞍点问题）, 以及并行化求解（基于PETSc)

建立了hybridized求解方式

## spyro
基于firedrake的全波形反演(FWI)的应用，见 [Geophysics/spyro](../../Geophysics/spyro/)

## geodynamics via Firedrake
地幔动力学模拟

## OpenTidalFarm
海洋潮汐能优化部署的应用程序

## 学习路线
```text
1. 安装firedrake库
2. 入门tutorial
Firedrake User Manual PDF, 2023

3. Introductory Jupyter notebooks
https://www.firedrakeproject.org/notebooks.html

4 Manual

5 Advanced tutorials

6 Thetis项目
```

## 参考文献
```text
Tuomas Karna, et al. 2018. Thetis coastal ocean model: discontinuous Galerkin discretization for the three-dimensional hydrostatic equations. Geosci. Model Dev., 11, 4359–4382. 

Keith J. Roberts, et al. 2022. spyro: a Firedrake-based wave propagation and full-waveform-inversion finite-element solver. Geosci. Model Dev., 15: 8639–8667. https://doi.org/10.5194/gmd-15-8639-2022

Thomas H. Gibson, et al. 2019. Slate: extending Firedrake’s domain-specific abstraction to hybridized solvers for geoscience and beyond. Geosci. Model Dev. Discuss., https://doi.org/10.5194/gmd-2019-86

D. Rhodri Davies, et al. 2022. Towards automatic finite-element methods for geodynamics via Firedrake. Geosci. Model Dev., 15: 5127–5166.

Thomas H. Gibson, Andrew T. T. McRae, Colin J. Cotter, Lawrence Mitchell, David A. Ham. Compatible Finite Element Methods for Geophysical Flows Automation and Implementation Using Firedrake. SpringerBriefs in Mathematics of Planet Earth  Weather, Climate, Oceans
```

## 相关文档

- [SurfaceWater](../)：学科入口
- [spyro](../../Geophysics/spyro/)：基于 Firedrake 的 FWI
- [Devito](../../Geophysics/Devito/)：有限差分 DSL（FWI 对照）
- [hpc-base/DSL](../../hpc-base/DSL/)：Firedrake / Devito 导读
- [hpc-base](../../hpc-base/)：PETSc、编译与并行环境
