# Underground

多孔介质流动与输移过程模拟（Darcy 尺度等）。已有许多开源程序，参见 Bilke 等（2019）的综述罗列，例如 MODFLOW、OPM 等：

```text
L. Bilke, B. Flemisch, T. Kalbacher, O. Kolditz, R. Helmig, T. Nagel,
Development of open-source porous media simulators: Principles and experiences,
Transp. Porous Media (2019)
```

目录名保留 `Underground`；主题涵盖地下水与油藏储层等。

## 子目录

- [porousMultiphaseFoam](./porousMultiphaseFoam/)：基于 OpenFOAM 的平面 2D/3D Richards 方程求解器
- [MODFLOW6](./MODFLOW6/)：USGS 地下水模拟系统；基于 PETSc 与 MPI 并行
- [OPM](./OPM/)：基于 DUNE、Zoltan 的黑油模型；开源后处理 ResInsight；可替代商业 ECLIPSE
- [DuMux-3.6](./DuMux-3.6/)：基于 DUNE 的多孔介质流体模拟；与 OPM 生态相关
- [GEOSX](./GEOSX/)：LLNL / Stanford / TotalEnergies / Chevron 等开发的地质碳封存等地下能源系统模拟
- [OpenGeoSys](./OpenGeoSys/)：THMC 模拟系统（ogs5 / ogs6）
- [waiwera](./waiwera/)：基于 MPI 与 PETSc 的地热及 CO? 封存模拟
- [golem](./golem/)：基于 MOOSE 框架的并行地热模拟
- [MPLBM-UT](./MPLBM-UT/)：格子 Boltzmann（palabos）后端，CT 岩石图像输入，毛细压力与相对渗透率等
- [LBDEMcoupling](./LBDEMcoupling/)：Palabos 与 LIGGGHTS 耦合，床面泥沙起动与水沙两相介观模型
- [LBPM](./LBPM/)：格子 Boltzmann 多孔介质相关
- [PFLOTRAN-OGS](./PFLOTRAN-OGS/)：PFLOTRAN 与 OGS 相关安装与文档

## 相关文档

- [Hydrology](../Hydrology/)：GSFLOW / pywatershed / ParFLOW 等与地下水耦合
- [Meshing](../Meshing/)：MODFLOW-USG / 角点网格前处理
- [Turbulence](../Turbulence/)：OpenFOAM 生态（与 porousMultiphaseFoam 相关）
- [hpc-base](../hpc-base/)：PETSc、MPI、DUNE 相关基座
- [GroundWater](../GroundWater/)：地下水专题入口（互补说明）
