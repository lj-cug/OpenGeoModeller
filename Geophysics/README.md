# Geophysics

地球物理的模拟程序，分为正演和反演模型；数值方法主要采用有限差分法和有限单元法。

## 子目录

- [first-arrival-time-pick](./first-arrival-time-pick/)：若干 Python GUI 初至拾取程序；拾取结果可作为 pyGIMLi 层析反演输入
- [pyGIMLi](./pyGIMLi/)：Python 地球物理反演；地震层析反演，可作为 FWI 初始速度模型
- [OpenSWPC-5.3.0](./OpenSWPC-5.3.0/)：交错结构网格下，有限差分法地震波传播正演
- [defmod-OpenSWPC](./defmod-OpenSWPC/)：地壳变形 defmod 与 OpenSWPC 耦合，研究地震波生成–传播全过程
- [Seissol](./Seissol/)：非结构网格下 ADER-DG 高阶间断 Galerkin 地震波正演
- [Devito](./Devito/)：有限差分特定域语言；与 JUDI 配合做 FWI
- [JUDI](./JUDI/)：Julia 全波形反演库（配合 Devito）
- [spyro](./spyro/)：基于 Firedrake 的间断 Galerkin / 自适应非结构网格 FWI
- [stride](./stride/)：基于 Devito 的医学成像全波形反演

## 相关文档

- [Meshing](../Meshing/)：SeismicMesh 等 FWI / 非结构网格
- [Geological-Modelling](../Geological-Modelling/)：地震解释与隐式建模
- [SurfaceWater/Firedrake](../SurfaceWater/Firedrake/)：Firedrake / DSL 基座（与 spyro 相关）
- [hpc-base](../hpc-base/)：MPI、编译与 DSL 相关基座文档
