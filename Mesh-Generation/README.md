# Mesh-Generation

地表水模型的前处理，即网格生成

商业软件Gambit等的非结构网格生成

开源软件，如GMSH等，的非结构网格生成

基于Python脚本的网格快速生成，以及不同格式网格文件之间的转换

## meshio

## GMSH

## JIGSAW

DARREN ENGWIRDA. Locally Optimal Delaunay-refinement and Optimisation-based Mesh Generation

## OCSMesh

非结构网格快速生成工具，针对SCHISM模型的网格文件格式(.gr3)

OCSMesh: a data-driven automated unstructured mesh generation software for coastal ocean modeling. NOAA Technical Memorandum NOS CS 47, 2021

## mesher

可根据地形变化、植被类型及其他网格优化要求（包括物理过程的特点），生成非结构化网格，针对分布式水文模型的三角网格生成

Christopher B. Marsh, et al. Multi-objective unstructured triangular mesh generation for use in hydrological and land surface models. Computers and Geosciences 119 (2018) 49–67

## OceanMesh2D

Keith J. Roberts, et al. OceanMesh2D 1.0: MATLAB-based software for two-dimensional unstructured mesh generation in coastal ocean modeling. Geosci. Model Dev., 12, 1847–1868, 2019

## 非结构网格编辑工具

### ACE Tools

SCHISM模拟自带的gr3格式的非结构网格的编辑工具，包括：设置边界条件等

### ADMESH

Colton J. Conroy, Ethan J. Kubatko, Dustin W. West, ADMESH: An advanced unstructured mesh generator for shallow water models, Ocean Dynamics, December 2012, Volume 62, Issue 10-12, pp 1503-1517

### BatTri

Ata Bilgili, et al. BatTri: A two-dimensional bathymetry-based unstructured triangular grid generator for finite element circulation modeling. Computers & Geosciences 32 (2006) 632–642

MATLAB GUI编程的三角形网格编辑工具，包括：区域切割、合并等操作

### CutMesh

### LaGridT

### 我自己编程的网格处理工具 (MyCode)

包括：地形插值、初始条件及边界条件设置、Gambit软件生成的中间网格文件格式转换、时间序列文件生成等

## High-Order-meshing

高阶数值格式，如DG法，的数值模拟的网格生成技术，是CAE领域的前沿课题

