# ShallowWater

浅水方程 (Shallow Water Equation) 的求解框架及模型。

## 子目录

- [TRITON](./TRITON/)：CPU / GPU 集群并行的结构网格洪水模拟
- [Volna-OP2](./Volna-OP2/)：基于 DSL OP2 的非结构网格海啸波传播
- [Titan2d](./Titan2d/)：基于 MPI 的滑坡模拟
- [BASEMENT](./BASEMENT/)：洛桑联邦理工学院非结构网格浅水求解器；基于 DSL-OP2，有界面；免费但不开源
- [amatos](./amatos/)：自适应浅水相关（含 StormFlash2d / TsunaFlash2d 等笔记）
- [sam(oa)2](./sam(oa)2/)：自适应网格浅水求解框架
- [GLM](./GLM/)：湖泊模型及相关 R / GUI 工具安装

## 相关文档

- [SurfaceWater](../SurfaceWater/)：河口 / 海洋浅水相关模式（ADCIRC、SCHISM 等）
- [Meshing](../Meshing/)：非结构网格前处理
- [hpc-base](../hpc-base/)：CUDA、MPI、OP2 / DSL 相关基座
- [Turbulence](../Turbulence/)：OpenFOAM 浅水求解器对照
