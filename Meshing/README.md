# Meshing

地表水、水文模型及地球物理模型的网格生成，主要采用 3 种方式（推荐第 3 种，工作效率最高）：

1. 商业软件，如 Gambit、SMS 等，生成非结构网格
2. 开源软件，如 Gmsh 等，生成非结构网格
3. 基于 Python 脚本的网格快速生成，以及不同格式网格文件之间的转换（利用 meshio）

## 子目录

- [Corner-Point-Mesh](./Corner-Point-Mesh/)：油气藏角点网格 (CPM)；Petrel 等可生成
- [Gambit](./Gambit/)：Gambit 网格格式转换与前处理（地形插值、初边值、时间序列等）
- [SMS](./SMS/)：地表水模拟系统 (SMS) 网格划分
- [Gmsh](./Gmsh/)：开源非结构网格工具（含 pygmsh 等）
- [OCSMesh](./OCSMesh/)：面向 SCHISM / ADCIRC / DGSWE 等 `.gr3` 的 Python 网格生成（JIGSAW 引擎）
- [mesher](./mesher/)：面向分布式水文的三角网格，可按地形 / 植被等优化
- [meshio](./meshio/)：多格式网格读写与转换
- [OceanMesh2D](./OceanMesh2D/)：Matlab 海洋非结构网格生成
- [SeismicMesh](./SeismicMesh/)：面向 FWI / spyro 的三角形非结构网格
- [MODFLOW-usg](./MODFLOW-usg/)：MODFLOW-USG 的 gridgen / ModelMuse 等
- [ACE-tools](./ACE-tools/)：SCHISM / SELFE 的 `.gr3` 编辑工具
- [BatTri](./BatTri/)：基于 bathymetry 的三角网格生成与编辑（MATLAB）
- [High-Order-Mesh](./High-Order-Mesh/)：高阶（如 DG）网格技术
- [Domain-Decomposition](./Domain-Decomposition/)：区域分解（METIS、SCOTCH、Zoltan 等）
- [fvcom.tbx](./fvcom.tbx/)：与 FVCOM 相关的网格工具

## 其他网格软件（参考）

```text
SALOME
FreeCAD
Blender
MeshLab
cfMesh
enGrid
Gmsh
Triangle
TetGen
NETGEN
```

## 非结构网格编辑工具

非结构网格生成后，可能需要编辑（切割、合并等）以及优化。

### ACE Tools

SCHISM 模拟自带的 `.gr3` 非结构网格编辑工具，包括设置边界条件等。见 [ACE-tools](./ACE-tools/)。

### ADMESH

Colton J. Conroy, Ethan J. Kubatko, Dustin W. West, ADMESH: An advanced unstructured mesh generator for shallow water models, Ocean Dynamics, December 2012, Volume 62, Issue 10-12, pp 1503-1517

### BatTri

Ata Bilgili, et al. BatTri: A two-dimensional bathymetry-based unstructured triangular grid generator for finite element circulation modeling. Computers & Geosciences 32 (2006) 632–642

MATLAB GUI 三角形网格编辑工具，包括区域切割、合并等。见 [BatTri](./BatTri/)。

### CutMesh

从大背景网格中裁剪小区域三角网格。

### High-Order-meshing

高阶数值格式（如 DG）的网格生成技术。见 [High-Order-Mesh](./High-Order-Mesh/)。

## 非结构网格生成经验

使用非结构网格模型，如 FVCOM、ADCIRC、SELFE、SCHISM 等，都需要生成三角形或其他非结构化网格（如 SCHISM 的 `hgrid.gr3`）。

Schneiders 罗列了大量网格生成软件：

<http://www.robertschneiders.de/meshgeneration/software.html>

### 非结构网格的编辑

为适应海洋数值模拟特点，可使用 ACE Tools (Turner and Baptista, 1999)、BatTri (Bilgili et al., 2006) 等。

- ACE Tools 针对 SELFE、SCHISM 输入格式，C 语言；调用 Triangle 生成网格，但主要是网格编辑。
- BatTri 用 MATLAB 调用 Triangle，生成快、编辑功能多，经验性参数较多，交互性强。

BatTri 以及 PIHM 可调用 Triangle 生成网格。ACE 和 BatTri 的主要功能体现在网格编辑；网格量级很大时操作较困难。

TetGen：

```text
Hang Si (2015). TetGen, a Delaunay-Based Quality Tetrahedral Mesh Generator.
ACM Trans. on Mathematical Software. 41 (2), Article 11 (February 2015), 36 pages.
```

### 非结构网格生成

1. Gambit、SMS，结合几何建模软件（AutoCAD、ArcGIS），可生成复杂区域三角或四边形网格，易操作，可到百万量级单元，并可设置边界条件。
2. 开源 Gmsh、TRIANGLE：Triangle 只需 PSLG（边界线段），效率高、易控制，可一次生成百万量级三角网格；GMSH 可生成复杂几何边界的 2D/3D 非结构网格。
3. MATLAB 网格程序便于使用、可重复性好，但百万量级时较慢。
4. Fortran / C 效率高，故 Triangle（C）生成效率很高。

DistMesh (Persson and Strang, 2004)、KMG (Koko, 2015)、OceanMesh2D 等：OceanMesh2D 用户介入少、可重复性好，需 shapefile 范围与 DEM；可视为 Gambit/SMS + ACE Tools 的一体化；适合近海局部加密，但 MATLAB 效率限制下建议网格量级约十万以下。

DistMesh、KMG 等适合量级较小（&lt;10?）且边界较简单的教学与算法演示。

### 网格优化

André F. 编写的 nicegrid2：输入格式面向 ADCIRC 和 SCHISM；可自动增减单元、降低歪斜度，并减少节点连接数，从而降低内存与计算量。

Aron Roland 的 polymesh：基于 Triangle，多为 FORTRAN，含误差预估与自适应网格功能。

### Domain-Decomposition

非结构网格区域分解库，常用 METIS、SCOTCH、Zoltan 等。见 [Domain-Decomposition](./Domain-Decomposition/)。

## 相关文档

- [SurfaceWater](../SurfaceWater/)：SCHISM、ADCIRC、DGSWE 等对 `.gr3` 的依赖
- [Hydrology](../Hydrology/)：PIHM / SHUD / mesher 三角网格
- [Geophysics](../Geophysics/)：SeismicMesh / spyro
- [Underground](../Underground/)：MODFLOW-USG / 角点网格
- [hpc-base](../hpc-base/)：编译与依赖库
