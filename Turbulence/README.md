# Turbulence

高精度湍流与 CFD 相关模型与工具。

## 子目录

- [GOTM](./GOTM/)：General Ocean Turbulence Model
- [OpenFOAM](./OpenFOAM/)：OpenCFD 框架及大量专用求解器 / 应用
- [DAFOAM](./DAFOAM/)：基于 CFD 与离散伴随的体型优化（同类：ADFLOW、SU2）
- [libAcoustics](./libAcoustics/)：基于 OpenFOAM 的气动声学模型
- [SU2](./SU2/)：斯坦福大学非结构网格气动优化 CFD 求解器（Windows / Linux / macOS）
- [HiFiLES](./HiFiLES/)：非结构网格高阶 LES，可跑 GPU 集群（已基本停止，团队转向 FR）
- [Wind_Energy](./Wind_Energy/)：风机 / 水轮机等相关 OpenFOAM 应用

## OpenFOAM 生态说明

在 OpenFOAM 框架上常见方向包括：

1. 基于 PDE 约束和伴随方法的优化
2. 气动声学
3. 多相流模型
4. 分子动力学模型
5. 精细湍流模型（LES、DNS）
6. 水轮机和风力发电机模拟（见 [Wind_Energy](./Wind_Energy/)）

注：ADFLOW 使用可压缩 RANS 与多块结构网格，适于机翼气动优化；SU2 为 C++、非结构网格、工作流完整；DAFOAM 以 OpenFOAM 为求解器，功能面广，且用 Python，较易使用。

[UniCFD Web-laboratory](https://github.com/unicfdlab)（ISP RAS）推动相关开源软件发展。

历史笔记中还提到过 ShallowFOAM、demFoam、sediFOAM、mdFOAM、fastFlume、finesed3d、tudflow3d 等方向，可作为检索关键词对照上游项目。

## 相关文档

- [Underground](../Underground/)：porousMultiphaseFoam 等 OpenFOAM 多孔介质求解器
- [ShallowWater](../ShallowWater/)：浅水求解器对照
- [Hemodynamics](../Hemodynamics/)：心血管 CFD
- [hpc-base](../hpc-base/)：编译、MPI、外部求解器
- [VirtualReality](../VirtualReality/)：后处理可视化
