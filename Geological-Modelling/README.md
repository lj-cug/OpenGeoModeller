# Geological-Modelling

地震数据解释与隐式地质建模相关工具与笔记。

## 地震数据解释

测井解释和地震解释的基本原理、软件与自动化工作流。

### 子目录

- [OpendTect6.6](./OpendTect6.6/)：地震解释；特色含并行计算与批处理（C++）；含实战与手册方向笔记
- [GMT](./GMT/)：地球科学常用制图程序；可在 OpendTect 中使用
- [Madagascar](./Madagascar/)：可重复开源地震数据处理（如 V4.1）；可在 OpendTect GUI 中调用
- [JTK](./JTK/)：科罗拉多矿业学院 / 中科大相关 Java 断层与层位解释工具（多线程）；含 OSV、MHE 等方向
- [AVO](./AVO/)：AVO 技术及 Python 程序说明

## 隐式地质建模

使用开源隐式地质建模 Python 工具，作为多孔介质流动模型的前处理：基于地震解释得到的断层和层位做隐式建模。

多数商业软件（如 Petrel、SKUA-GOCAD）偏显式建模，需大量手工构建地质体；隐式建模主要依靠梯度 / dip-azimuth 等属性势场插值，手工介入少，但对内存与算力要求较高。当前主要工具：GemPy、LoopStructural。

### 子目录

- [GemPy](./GemPy/)：亚琛工业大学隐式地质建模 Python 程序
- [LoopStructural](./LoopStructural/)：澳大利亚隐式地质建模 Python 程序
- [SKUA-GOCAD](./SKUA-GOCAD/)：商业地质建模软件，可与上述工具配合

角点网格 (CPM) 数据结构与格式转换参见 [Meshing/Corner-Point-Mesh](../Meshing/Corner-Point-Mesh/)。

## 相关文档

- [Geophysics](../Geophysics/)：正演 / 反演与 FWI
- [Meshing](../Meshing/)：角点网格与前处理
- [Underground](../Underground/)：地质建模结果作为多孔介质模拟输入
- [hpc-base](../hpc-base/)：并行与语言环境
