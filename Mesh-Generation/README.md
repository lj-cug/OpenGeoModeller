# Mesh-Generation

[�ǽṹ�������ɼ��༭](./�ǽṹ�������ɼ��༭.md)

�ر�ˮ��ˮ��ģ�ͼ���������ģ�͵��������ɣ�������

1. ��ҵ���Gambit�ȵķǽṹ��������

2. ��Դ�������GMSH�ȣ��ķǽṹ��������

3. ����Python�ű�������������ɣ��Լ���ͬ��ʽ�����ļ�֮���ת��

## GMSH

## JIGSAW

DARREN ENGWIRDA. Locally Optimal Delaunay-refinement and Optimisation-based Mesh Generation

## OCSMesh

�ǽṹ����������ɹ��ߣ����SCHISMģ�͵������ļ���ʽ(.gr3)

OCSMesh: a data-driven automated unstructured mesh generation software for coastal ocean modeling. NOAA Technical Memorandum NOS CS 47, 2021

## mesher

�ɸ��ݵ��α仯��ֲ�����ͼ����������Ż�Ҫ�󣨰���������̵��ص㣩�����ɷǽṹ��������Էֲ�ʽˮ��ģ�͵�������������

Christopher B. Marsh, et al. Multi-objective unstructured triangular mesh generation for use in hydrological and land surface models. Computers and Geosciences 119 (2018) 49�C67

## OceanMesh2D

Keith J. Roberts, et al. OceanMesh2D 1.0: MATLAB-based software for two-dimensional unstructured mesh generation in coastal ocean modeling. Geosci. Model Dev., 12, 1847�C1868, 2019

## SeismicMesh

��������ȫ���η���(FWI)ʹ�õ������ηǽṹ����

## MODFLOW-usg-Mesh

����MODFLOW-usg�ķǽṹ�������ɹ���gridgen�ͺ�����ModelMuse

# �ǽṹ����༭����

## ACE Tools

SCHISMģ���Դ���gr3��ʽ�ķǽṹ����ı༭���ߣ����������ñ߽�������

## ADMESH

Colton J. Conroy, Ethan J. Kubatko, Dustin W. West, ADMESH: An advanced unstructured mesh generator for shallow water models, Ocean Dynamics, December 2012, Volume 62, Issue 10-12, pp 1503-1517

## BatTri

Ata Bilgili, et al. BatTri: A two-dimensional bathymetry-based unstructured triangular grid generator for finite element circulation modeling. Computers & Geosciences 32 (2006) 632�C642

MATLAB GUI��̵�����������༭���ߣ������������и�ϲ��Ȳ���

## CutMesh

## LaGridT

## Gambit-Post

���Լ�ʹ��FORTRAN��̵����ڴ�����ҵ���Gambit���������ļ��ĸ�ʽת�������������β�ֵ����ʼ�������߽��������á�Gambit������ɵ��м������ļ���ʽת����ʱ�������ļ����ɵ�

## OCSMesh

SCHISM�ķǽṹ�������ɵ�Python����

# High-Order-meshing

�߽���ֵ��ʽ����DG��������ֵģ����������ɼ�������CAE�����ǰ�ؿ���

# meshio

�����ǽṹ�����ļ�֮��ĸ�ʽת��Python�ű�

# �ǽṹ�������ɵ�һЩ�����ܽ�

ʹ�÷ǽṹ����ģ�ͣ���FVCOM, ADCIRC, SELFE,
SCHISM�ȣ�����Ҫ���������λ��������͵ķǽṹ��������SCHISMģ����Ҫhgrid.gr3�ļ���

�ǽṹ������������ܶ࣬Schneiders���������վ�����˺ܶࡣ

http://www.robertschneiders.de/meshgeneration/software.html

## �ǽṹ����༭

Ϊ��Ӧ������ֵģ���һЩ�ص㣬��Ҫ�����ɵ�������б༭����ʹ��ACE Tools
(Turner and Baptista, 1999), BatTri (Bilgili et al., 2006), ...

-   ACE
    Toolsר�����SELFE��SCHISMģ�͵������ļ���ʽ�����ģ�����C���Ա�̣�ACE
    Tools����Triangle�������񣬵���Ҫ������༭��

-   BatTriʹ��MATLAB�������Triangle���ĳ�����ˣ����������ٶȺܿ죬�Ҿ��кܶ�����༭���ܣ����༭����ľ����Բ����϶ࡣBatTri����༭�Ľ����Ժ�ǿ��������ѡ����ʾ��

BatTri�Լ�PIHM�ɵ���Triangle�����������񣬼���Ч�ʸߡ�

ACE��BatTri����Ҫ��������������༭�ϡ��������������ϴ�ʱ�������Ƚ����ѡ�

CutMesh, �Ӵ�ı��������У��ü�һ��С�������������

## �ǽṹ��������

��1��Gambit, SMS����ϼ��ν�ģ�����AutoCAD,
ArcGIS)�����Է�������ɸ�����������ǻ��ı����������ڲ����������ɰ���������Ԫ�ķǽṹ������ͬʱ�������ñ߽������ȣ�����ǿ��

��2����Դ�����GMSH, TRIANGLE:
Triangleֻ��ҪPSLG�ļ����߽��߶Σ�����������Ч����ߣ����ڿ��ƣ���һ�������ɰ���������Ԫ����������

GMSH�����ɸ��Ӽ��α߽��2D/3D�ǽṹ������

��3��MATLAB������������ɳ���ܷ���ʹ�ã��ҿ��������������ظ��ԺͿɲ����ԣ���������Ԫ�������ϴ�ʱ���ﵽ���򣩣�������̺�����

��4��Fortran��C���Եļ���Ч�ʽϸߣ���ˣ�Triangle��C���ԣ�����������Ч�ʺܸߡ�

DistMesh(Persson and Strang, 2004), KMG (Koko, 2015), OceanMesh2D ()

OceanMesh2D��Ҫ�û�������٣���������Ĳ������ظ��Ժܺã�����Ҫ�����������Χ(shapefile)�͵�������(DEM)����������DistMesh���㷨��OceanMesh2D������������༭��һ�壬����OceanMesh2D
= Gambit/SMS + ACE
Tools���ɲ����ԺͿ��ظ���ǿ������к����ߵ�shapefile�ļ���DEM��nc�ļ����ǳ���������������Խ�������ģ�⣬�ɷ����������ֲ��ֱ��ʡ�����MATLAB����Ч�ʽϵͣ�������������Ӧ��ʮ�����¡�

��DistMesh,
KMG�ȣ�����MATLAB���Ա�̣��ɷ�������һЩ������С��\<10^4^)�ҽ�ϱ߽�ϼ򵥵����񡣿����ڷǽṹ���������ɼ���ר���о��ͽ�ѧ��ʾ��

## �����Ż�

Andr�� F��nicegrid2:
�ǽṹ������༭����FORTRAN���ԣ����Զ����Ӻͼ�������Ԫ����˽����������б��(skewness)��ͬʱ���ɼ��ٽڵ���Χ���ӽڵ����Ŀ�����������������ģ��ģ����������Ĺ�ģ�������ڴ���Ҫ�ͼ�������

Aron Roland��polymesh:
����Triangle���󲿷�ΪFORTRAN���룬�Լ���Triangle�Ľӿڳ����Ҿ������Ԥ��������Ӧ����Ĺ��ܡ�