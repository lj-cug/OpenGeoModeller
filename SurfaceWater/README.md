# SurfaceWater

地表水模拟系统，指模拟河流、湖泊和海洋等地表水体流动和物质输移，如波浪、泥沙输移和河（海）床演变、污染物输移和水质过程等。

## 子目录

- [SCHISM](./SCHISM/)：三维，半隐格式，有限元，美国弗吉尼亚海洋研究所 VIMS
- [shyfem](./shyfem/)：三维，半隐格式，有限单元法，高性能求解器，意大利
- [ADCIRC](./ADCIRC/)：平面二维，风暴潮，美国圣母大学开发
- [SUNTANS](./SUNTANS/)：三维，有限体积法，质量守恒，美国斯坦福大学
- [telemac](./telemac/)：平面二维，线性有限元，法国电力公司 (EDF)
- [DGSWE](./DGSWE/)：平面二维，间断 Galerkin 法，美国
- [WaveWatch-III](./WaveWatch-III/)：波浪模型，美国 / 中国台湾
- [CROCO](./CROCO/)：AMR 技术的海洋动力学模型，法国
- [NEMO](./NEMO/)：欧洲开发的 3D 海洋动力学模型，法国
- [SLIM3D](./SLIM3D/)：DG 法的 3D 海洋动力学模式，比利时
- [MITGcm](./MITGcm/)：较简单的 GCM 模式，常在 ESM 中使用，美国
- [Delft3D-FM](./Delft3D-FM/)：非结构网格版本的 Delft3D-FLOW，用于计算水力学教学
- [Firedrake](./Firedrake/)：基于 DSL 的 DG 有限单元法开发库，英国帝国理工大学
- [ROMS](./ROMS/)：区域海洋模式

## 相关文档

- [Meshing](../Meshing/)：非结构网格前处理（如 `.gr3`）
- [ShallowWater](../ShallowWater/)：浅水 / 洪水相关求解器
- [ESM-Coupler](../ESM-Coupler/)：与大气 / 波浪等耦合
- [hpc-base](../hpc-base/)：编译器、MPI、PETSc、容器等
- [VirtualReality](../VirtualReality/)：后处理与原位可视化
