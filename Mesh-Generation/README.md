# Mesh-Generation

地表水、水文模型及地球物理模型的网格生成，主要采用3种方式(推荐第3种,工作效率最高)：

1. 商业软件，如Gambit, SMS等，生成非结构网格

2. 开源软件，如GMSH等，生成非结构网格

3. 基于Python脚本的网格快速生成，以及不同格式网格文件之间的转换(利用meshio工具)

## Corner-Point-Mesh

油气藏模拟器中常用的网格类型：角点网格(Corner-Point Mesh),Petrel软件可生成CPM网格，是油气藏模拟的前处理程序

## Gambit

我自己使用FORTRAN编程的用于处理商业软件Gambit生成网格文件的格式转换，包括：地形插值、初始条件及边界条件设置、Gambit软件生成的中间网格文件格式转换、时间序列文件生成等

## SMS

地表水模拟系统(SMS)的网格划分软件，在工程应用中拥有很多用户

## GMSH

开源的非结构网格划分工具，有很多强大的工具，如pygmsh

## OCSMesh

针对SCHISM, ADCIRC, DGSWE等模型的网格文件格式(.gr3)的非结构网格生成的Python程序，使用JIGSAW作为网格生成引擎

## LaGridT


## mesher

可根据地形变化、植被类型及其他网格优化要求，生成非结构化网格，针对分布式水文模型的三角网格生成

## OceanMesh2D

Matlab语言编写的海洋模式非结构网格生成工具，计算效率不高

## SeismicMesh

生成用于全波形反演(FWI)使用的三角形非结构网格，用于有限单元法的spyro模型

## MODFLOW-usg-Mesh

用于MODFLOW-usg的非结构网格生成工具gridgen和后处理工具ModelMuse

## 其他

```
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

# 非结构网格编辑工具

非结构网格生成后，可能根据工程情况，需要对其进行编辑(如切合,合并等)以及优化

## ACE Tools

SCHISM模拟自带的gr3格式的非结构网格的编辑工具，包括：设置边界条件等

## ADMESH

Colton J. Conroy, Ethan J. Kubatko, Dustin W. West, ADMESH: An advanced unstructured mesh generator for shallow water models, Ocean Dynamics, December 2012, Volume 62, Issue 10-12, pp 1503-1517

## BatTri

Ata Bilgili, et al. BatTri: A two-dimensional bathymetry-based unstructured triangular grid generator for finite element circulation modeling. Computers & Geosciences 32 (2006) 632–642

MATLAB GUI编程的三角形网格编辑工具，包括：区域切割、合并等操作

## CutMesh

## High-Order-meshing

高阶数值格式，如DG法，的数值模拟的网格生成技术


# 非结构网格生成经验

使用非结构网格模型，如FVCOM, ADCIRC, SELFE,
SCHISM等，都需要生成三角形或其他类型的非结构化网格，如SCHISM模型需要hgrid.gr3文件。

非结构网格生成软件很多，Schneiders在下面的网站罗列了很多。

http://www.robertschneiders.de/meshgeneration/software.html

## 非结构网格的编辑

为适应海洋数值模拟的一些特点，需要对生成的网格进行编辑，可使用ACE Tools
(Turner and Baptista, 1999), BatTri (Bilgili et al., 2006), ...

-   ACE
    Tools专门针对SELFE、SCHISM模型的输入文件格式开发的，采用C语言编程，ACE
    Tools调用Triangle生成网格，但主要是网格编辑。

-   BatTri
	使用MATLAB程序调用Triangle核心程序。因此，生成网格速度很快，且具有很多网格编辑功能，但编辑网格的经验性参数较多。
	BatTri网格编辑的交互性很强，有选项提示。

BatTri以及PIHM可调用Triangle程序生成网格，计算效率高。

ACE和BatTri的主要功能体现在网格编辑上。当网格数量级较大时，操作比较困难。

CutMesh, 从大的背景网格中，裁剪一块小区域的三角网格。

TetGen
```
Hang Si (2015). TetGen, a Delaunay-Based Quality Tetrahedral Mesh Generator. ACM Trans. on Mathematical Software. 41 (2), Article 11 (February 2015), 36 pages.
```

## 非结构网格生成

（1）Gambit, SMS，结合几何建模软件（AutoCAD,
ArcGIS)，可以方便地生成复杂区域的三角或四边形网格，易于操作，可生成百万量级单元的非结构化网格。同时，可设置边界条件等，功能强大。

（2）开源代码的GMSH, TRIANGLE:
Triangle只需要PSLG文件（边界线段），生成网格效率最高，易于控制，可一次性生成百万量级单元的三角网格。

GMSH可生成复杂几何边界的2D/3D非结构化网格。

（3）MATLAB程序的网格生成程序很方便使用，且可提高生成网格的重复性和可操作性，但当网格单元数量级较大时（达到百万），计算过程很慢。

（4）Fortran和C语言的计算效率较高，因此，Triangle（C语言）的网格生成效率很高。

DistMesh(Persson and Strang, 2004), KMG (Koko, 2015), OceanMesh2D ()

OceanMesh2D需要用户介入较少，生成网格的操作可重复性很好，仅需要定义计算区域范围(shapefile)和地形数据(DEM)，并吸收了DistMesh的算法。
OceanMesh2D集网格生成与编辑于一体，即：OceanMesh2D = Gambit/SMS + ACE
Tools。
OceanMesh2D可操作性和可重复性强。如果有海岸线的shapefile文件和DEM的nc文件，非常方便生成网格。针对近海区域模拟，可方便控制网格局部分辨率。但是，由于MATLAB计算效率较低，生成网格量级应在十万以下。

如DistMesh,
KMG等，采用MATLAB语言编程，可方便生成一些量级较小（\<10^4^)且结合边界较简单的网格。可用于非结构化网格生成技术专门研究和教学演示。

## 网格优化

André F.编写的nicegrid2:

网格输入格式： ADCIRC和SCHISM模式

非结构化网格编辑程序，FORTRAN语言，可自动增加和减少网格单元，因此降低网格的歪斜度。同时，可减少节点周围连接节点的数目，这样减少了网格规模和模型质量矩阵的规模，降低内存需要和计算量。

Aron Roland的polymesh:

基于Triangle，大部分为FORTRAN代码，以及与Triangle的接口程序，且具有误差预估和自适应网格的功能。

