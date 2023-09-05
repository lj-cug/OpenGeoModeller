# ParaView-Python程序目录及介绍

## [PVGeo]{.mark}

https://pvgeo.org/index.html

[地球可视化(Geovisualization)]{.mark}

PVGeo解决地球科学数据可视化软件兼容性问题，着眼于[更顺畅和更直接地]{.mark}将地球科学（固体地球物理）数据输入基于VTK的软件环境，如ParaView，而避免繁重的重复性开发VTK软件插件。

PVGeo连接地球科学数据与基于VTK的3D渲染环境，方便于特征分析，如体渲染、glyphing,
subsetting, K-Means clustering, 体插值,
iso-contouring和VR。这样，地球科学家就可以控制所有ParaView和其他基于VTK的库，如ParaViewWeb,VTK.js,
PyVista(Sullivan & Kaszynski, 2019)或者将数据扩展至新领域，如VR。

![](./media/image1.emf)

图1 PyGeo将地球科学与VTK和ParaView连接用于数据可视化

### 简介 {#简介 .标题3}

PVGeo是基于PyVista，PVGeo提供一个扩展包到PyVista，连接数据格式和地球科学中常见的过滤子程序，到PyVista的3D可视化框架。PVGeo使用PyVista使PVGeo算法的输入和输出变得更容易，用户[可重复的]{.mark}工作流程整合可视化任务。

Sullivan et al. 2019. PVGeo: an open-source Python package for
geoscientific visualization in VTK and ParaView. Journal of Open Source
Software, 4(38), 1451. <https://doi.org/10.21105/joss.01451>

## [PyVista]{.mark}

科学数据可视化的软件已有很多，诸如有名的Matplotlib(Hunter, 2007), Mayavi
(Ramachandran & Varoquaux, 2011)、yt项目(Turk et al.,
2010)以及Visualization Toolkit (VTK) (Schroeder, Lorensen, & Martin,
2006)。

其中，VTK-Python融合了C++的计算速度和Python的快速建模的优势。但是，使用VTK-Python处理地球科学数据还是或[涉及复杂的绑定]{.mark}API的使用问题。

PyVista即可解决上述问题。

PyVista封装了很多VTK库的常用算法，提供了一套共享的功能集。其核心是纯Python帮助木块，通过Numpy返回VTK数据和VTK的面向对象方法直接数组访问，实现3D可视化(Schroeder
et al., 2006)。

![](./media/image2.emf)

> 图1
> PyVista可视化地球科学数据的例子，渲染图包括：数字化陆地表面并使用卫星影像和地质图覆盖，地表下温度模型、采样的温度值散点和地球物理钻井日志数据、GIS地点边界和解译的断层表面。

Sullivan, C. B., & Kaszynski, A. 2019. PyVista: 3D plotting and mesh
analysis through a streamlined interface for the visualization toolkit
(VTK). Journal of Open Source Software, 4(37), 1450.
doi:10.21105/joss.01450

## [pv_atmos]{.mark} 

用于4D可视化（3D空间+时间）以NetCDF格式存储的大气数据（以及海洋数据）。使用Python2脚本(ver.\>2.5.6))和ParaView(ver.\>=4.1)加载、处理和可视化数据。脚本可自动加载经度-纬度-压力网格、再计算压力对数网格或球坐标。还可以添加网格线、平面和标记等。有2个示例脚本。

![](./media/image3.emf)

Jucker, M 2014 Scientific Visualisation of Atmospheric Data with
ParaView. Journal of Open Research Software, 2(1): e4, DOI:
http://dx.doi.org/10.5334/jors.al

## [pyside2-embed-paraview]{.mark}

试验性项目。将ParaView渲染窗口内嵌入PySide2 GT
GUI。目的是：探讨将ParaView窗口嵌入Qt应用程序的可行性（通过PySider2）,然后通过已有的Paraview/Python接口API控制Qt应用程序。

### 要求

Qt 5.9.1

Python 2.7

PySide2：

\- Python 2.7 bindings to Qt5. Provides shiboken2, which will be used to

generate a Qt widget library accessible to Python.

## Paraview-to-POVRay-Water-Render

A python script to export isosurfaces from ParaView and render a scene
with [natural water appeareance]{.mark} by using Ray Tracing technique.

将等值面输出，然后使用光追技术将其渲染成自然水面的样子！

需要的软件

ParaView (https://www.paraview.org/download/)

POV-Ray (https://www.povray.org/download/)

FFmpeg (<https://www.ffmpeg.org/download.html>)

## paraview-AMR_movie

自适应网格AMR的可视化，并做动画演示。

## ParaViewConnect

Python library to help starting and connecting to remote ParaView
servers

Ensure you have Python 2.7 (including virtualenv package) and Paraview
installed.

[Note ParaView needs to use the same version of python]{.mark}

## paraview-amrclaw

Tests for the eventual use of paraview with geoclaw/amrclaw for 3D
visualization

## pv_utils-Tecplot

Utilities to facilitate creating 3D images from .tec files in Paraview

## cmocean-paraview

海洋的颜色条文件，放入paraview的相关路径！
