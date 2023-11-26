# Mesh-Generation

�ر�ˮ��ˮ��ģ�ͼ���������ģ�͵��������ɣ���Ҫ����3�ַ�ʽ(�Ƽ���3��,����Ч�����)��

1. ��ҵ�������Gambit, SMS�ȣ����ɷǽṹ����

2. ��Դ�������GMSH�ȣ����ɷǽṹ����

3. ����Python�ű�������������ɣ��Լ���ͬ��ʽ�����ļ�֮���ת��(����meshio����)

## Corner-Point-Mesh

������ģ�����г��õ��������ͣ��ǵ�����(Corner-Point Mesh),Petrel���������CPM������������ģ���ǰ�������

## Gambit

���Լ�ʹ��FORTRAN��̵����ڴ�����ҵ���Gambit���������ļ��ĸ�ʽת�������������β�ֵ����ʼ�������߽��������á�Gambit������ɵ��м������ļ���ʽת����ʱ�������ļ����ɵ�

## SMS

�ر�ˮģ��ϵͳ(SMS)�����񻮷�������ڹ���Ӧ����ӵ�кܶ��û�

## GMSH

��Դ�ķǽṹ���񻮷ֹ��ߣ��кܶ�ǿ��Ĺ��ߣ���pygmsh

## OCSMesh

���SCHISM, ADCIRC, DGSWE��ģ�͵������ļ���ʽ(.gr3)�ķǽṹ�������ɵ�Python����ʹ��JIGSAW��Ϊ������������

## LaGridT


## mesher

�ɸ��ݵ��α仯��ֲ�����ͼ����������Ż�Ҫ�����ɷǽṹ��������Էֲ�ʽˮ��ģ�͵�������������

## OceanMesh2D

Matlab���Ա�д�ĺ���ģʽ�ǽṹ�������ɹ��ߣ�����Ч�ʲ���

## SeismicMesh

��������ȫ���η���(FWI)ʹ�õ������ηǽṹ�����������޵�Ԫ����spyroģ��

## MODFLOW-usg-Mesh

����MODFLOW-usg�ķǽṹ�������ɹ���gridgen�ͺ�����ModelMuse

## ����

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

# �ǽṹ����༭����

�ǽṹ�������ɺ󣬿��ܸ��ݹ����������Ҫ������б༭(���к�,�ϲ���)�Լ��Ż�

## ACE Tools

SCHISMģ���Դ���gr3��ʽ�ķǽṹ����ı༭���ߣ����������ñ߽�������

## ADMESH

Colton J. Conroy, Ethan J. Kubatko, Dustin W. West, ADMESH: An advanced unstructured mesh generator for shallow water models, Ocean Dynamics, December 2012, Volume 62, Issue 10-12, pp 1503-1517

## BatTri

Ata Bilgili, et al. BatTri: A two-dimensional bathymetry-based unstructured triangular grid generator for finite element circulation modeling. Computers & Geosciences 32 (2006) 632�C642

MATLAB GUI��̵�����������༭���ߣ������������и�ϲ��Ȳ���

## CutMesh

## High-Order-meshing

�߽���ֵ��ʽ����DG��������ֵģ����������ɼ���


# �ǽṹ�������ɾ���

ʹ�÷ǽṹ����ģ�ͣ���FVCOM, ADCIRC, SELFE,
SCHISM�ȣ�����Ҫ���������λ��������͵ķǽṹ��������SCHISMģ����Ҫhgrid.gr3�ļ���

�ǽṹ������������ܶ࣬Schneiders���������վ�����˺ܶࡣ

http://www.robertschneiders.de/meshgeneration/software.html

## �ǽṹ����ı༭

Ϊ��Ӧ������ֵģ���һЩ�ص㣬��Ҫ�����ɵ�������б༭����ʹ��ACE Tools
(Turner and Baptista, 1999), BatTri (Bilgili et al., 2006), ...

-   ACE
    Toolsר�����SELFE��SCHISMģ�͵������ļ���ʽ�����ģ�����C���Ա�̣�ACE
    Tools����Triangle�������񣬵���Ҫ������༭��

-   BatTri
	ʹ��MATLAB�������Triangle���ĳ�����ˣ����������ٶȺܿ죬�Ҿ��кܶ�����༭���ܣ����༭����ľ����Բ����϶ࡣ
	BatTri����༭�Ľ����Ժ�ǿ����ѡ����ʾ��

BatTri�Լ�PIHM�ɵ���Triangle�����������񣬼���Ч�ʸߡ�

ACE��BatTri����Ҫ��������������༭�ϡ��������������ϴ�ʱ�������Ƚ����ѡ�

CutMesh, �Ӵ�ı��������У��ü�һ��С�������������

TetGen
```
Hang Si (2015). TetGen, a Delaunay-Based Quality Tetrahedral Mesh Generator. ACM Trans. on Mathematical Software. 41 (2), Article 11 (February 2015), 36 pages.
```

## �ǽṹ��������

��1��Gambit, SMS����ϼ��ν�ģ�����AutoCAD,
ArcGIS)�����Է�������ɸ�����������ǻ��ı����������ڲ����������ɰ���������Ԫ�ķǽṹ������ͬʱ�������ñ߽������ȣ�����ǿ��

��2����Դ�����GMSH, TRIANGLE:
Triangleֻ��ҪPSLG�ļ����߽��߶Σ�����������Ч����ߣ����ڿ��ƣ���һ�������ɰ���������Ԫ����������

GMSH�����ɸ��Ӽ��α߽��2D/3D�ǽṹ������

��3��MATLAB������������ɳ���ܷ���ʹ�ã��ҿ��������������ظ��ԺͿɲ����ԣ���������Ԫ�������ϴ�ʱ���ﵽ���򣩣�������̺�����

��4��Fortran��C���Եļ���Ч�ʽϸߣ���ˣ�Triangle��C���ԣ�����������Ч�ʺܸߡ�

DistMesh(Persson and Strang, 2004), KMG (Koko, 2015), OceanMesh2D ()

OceanMesh2D��Ҫ�û�������٣���������Ĳ������ظ��Ժܺã�����Ҫ�����������Χ(shapefile)�͵�������(DEM)����������DistMesh���㷨��
OceanMesh2D������������༭��һ�壬����OceanMesh2D = Gambit/SMS + ACE
Tools��
OceanMesh2D�ɲ����ԺͿ��ظ���ǿ������к����ߵ�shapefile�ļ���DEM��nc�ļ����ǳ���������������Խ�������ģ�⣬�ɷ����������ֲ��ֱ��ʡ����ǣ�����MATLAB����Ч�ʽϵͣ�������������Ӧ��ʮ�����¡�

��DistMesh,
KMG�ȣ�����MATLAB���Ա�̣��ɷ�������һЩ������С��\<10^4^)�ҽ�ϱ߽�ϼ򵥵����񡣿����ڷǽṹ���������ɼ���ר���о��ͽ�ѧ��ʾ��

## �����Ż�

Andr�� F.��д��nicegrid2:

���������ʽ�� ADCIRC��SCHISMģʽ

�ǽṹ������༭����FORTRAN���ԣ����Զ����Ӻͼ�������Ԫ����˽����������б�ȡ�ͬʱ���ɼ��ٽڵ���Χ���ӽڵ����Ŀ�����������������ģ��ģ����������Ĺ�ģ�������ڴ���Ҫ�ͼ�������

Aron Roland��polymesh:

����Triangle���󲿷�ΪFORTRAN���룬�Լ���Triangle�Ľӿڳ����Ҿ������Ԥ��������Ӧ����Ĺ��ܡ�

