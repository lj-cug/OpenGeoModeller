# Devito

有限差分法的特定域语言库，主要是离散地震波传播的声波方程，实现全波形反演(Full Waveform Inversion, FWI)。

编程语言： C, Python

## 本仓库文档

- [install](./install/)：Devito 安装
- [doc](./doc/)：基于 Devito 与 JUDI 的 FWI 操作
- [FWI-HPC软件调研20230512.md](./FWI-HPC软件调研20230512.md)

## 特色

快速实施有限差法(FDM)离散声波方程

有限单元法(FEM)离散PDE的DSL库可使用firedrake

## 参考文献

Mathias Louboutin, et al. Devito (v3.1.0): an embedded domain-specific language for finite differences and geophysical exploration.  Geosci. Model Dev., 12, 1165-1187, 2019

## 相关文档

- [Geophysics](../)：学科入口
- [JUDI](../JUDI/)：Julia 封装的 FWI / RTM
- [spyro](../spyro/)：Firedrake 路线的 FWI
- [SurfaceWater/Firedrake](../../SurfaceWater/Firedrake/)：有限元 DSL
- [hpc-base/DSL](../../hpc-base/DSL/)：Devito 导读
