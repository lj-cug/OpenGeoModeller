# spyro

采用非结构网格及waveform自适应网格技术，以及有限单元法，求解地震波的声波方程

实现全波形反演全流程，包括2D和3D FWI

基于firedrake库，采用Python语言编程

## 工作流程

1. 使用SeismicMesh，生成地震速度模型的有限单元非结构网格

2. 使用spyro执行2D/3D的全波形反演

## 参考文献

Keith J. Roberts, et al. spyro: a Firedrake-based wave propagation and full-waveform-inversion finite-element solver. Geosci. Model Dev., 15, 8639–8667, 2022

Roberts, K., Gioria, R., and Pringle, W.: SeismicMesh: Triangular meshing for seismology, JOSS, 6, 2687, https://doi.org/10.21105/joss.02687, 2021
