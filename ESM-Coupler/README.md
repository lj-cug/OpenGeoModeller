# ESM-Coupler

区域性地球耦合模拟系统，以及耦合器（对原始代码侵入度较低，实现不同模式组件的耦合）。耦合器包括 ESMF、BMI、OASIS3-MCT 等。

综述参考文献：

```text
Bert Jagers. Linking Data, Models and Tools: An Overview.
5th International Congress on Environmental Modelling and Software -
Ottawa, Ontario, Canada - July 2010
```

## 耦合器

- [ESMF](./ESMF/)：Earth System Modeling Framework
- [OASIS3-MCT](./OASIS3-MCT/)：OASIS3 与 MCT 耦合
- [BMI](./BMI/)：Basic Model Interface
- [OpenMI](./OpenMI/)：Open Modelling Interface

## 代表性 ESM

- [CESM](./CESM/)：全球尺度地球耦合模拟系统，HPC 社区广泛使用
- [RegESM](./RegESM/)：区域性地球耦合模拟；具备 Catalyst 在线可视化，观察热带风暴等快过程
- [SCRIPPS](./SCRIPPS/)：WRF-4.1.1 + MITgcm + WaveWatch-III
- [COAWST](./COAWST/)：USGS 近海岸环境区域性 ESM
- [MOSSCO](./MOSSCO/)：模块化海岸系统耦合
- [CoastalApp](./CoastalApp/)：近海岸模拟系统；ESMF 耦合多模式，可读 HWRF，定量模拟风暴潮
- [CLiMA](./CLiMA/)：气候建模相关组件与笔记

## 相关文档

- [Meteorology](../Meteorology/)：WRF / HWRF 等大气组件
- [SurfaceWater](../SurfaceWater/)：ROMS、CROCO、MITGcm、WW3 等海洋 / 波浪组件
- [VirtualReality](../VirtualReality/)：Catalyst / 原位可视化
- [hpc-base](../hpc-base/)：MPI、编译与依赖
