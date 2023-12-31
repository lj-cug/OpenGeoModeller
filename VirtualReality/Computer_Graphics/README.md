# Computer Graphics

计算机图形学基础以及一些基于CPU和GPU做光线追踪的图形渲染程序库

具体介绍见 ./doc/科学可视化与图形渲染的程序介绍20220210.docx

## OpenGL

CG基础库OpenGL的学习文档，以及CUDA-OpenGL交互的学习材料

## Off-screen Rendering

OSMesa Off-screen Rendering与ParaView的基于EGL的Off-screen图形渲染

## Ray-tracing Libraries

### POVRay

The Persistence of Vision Ray Tracer, or POV-Ray，基于CPU的光线追踪渲染库

### OSPRay

Intel公司研发的基于CPU的光线追踪渲染库

### Nvidia-IndeX和OptiX

Nvidia公司研发的基于GPU的光线追踪渲染库

### proland-4.0

法国INRIA研发的用于地球科学图形渲染的库

## VTK (Visualization ToolKit)

CFD常用的可视化图形库

## IceT

集群图形渲染库

参考：Haeyong Chung et al., A survey of Software frameworks for cluster-based large high-resolution displays  IEEE, 2014, 20(8)

## Blender

开源的图形渲染程序

## KML

Google Earth的图形语言

# 计算机图形渲染(光线追踪)

## POV-Ray （开源）

POV-Ray，全名是[Persistence of Vision
Raytracer]{.mark}，是一个使用光线跟踪绘制三维图像的[开放源代码](https://baike.baidu.com/item/%E5%BC%80%E6%94%BE%E6%BA%90%E4%BB%A3%E7%A0%81/114160)免费软件。

运行POV[脚本语言](https://baike.baidu.com/item/%E8%84%9A%E6%9C%AC%E8%AF%AD%E8%A8%80)。它是基于DKBTrace来开发的,
DKBTrace是由 David Kirk Buck和 Aaron A. Collins编写在 Amiga上的.
POV-ray早期也受到了Polyray raytracer 作者 Alexander Enzmann
的帮助。很多漂亮的图片就是由POV-ray来制作的。

该软件最初发展始于80年代。, David Kirk
Buck下载了一个为?[Unix](https://baike.baidu.com/item/Unix/219943)编写的[Amiga](https://baike.baidu.com/item/Amiga/10443049)光线跟踪软件的
source code .
有趣的是，当他玩了一段时间后，他觉得应该自己写一个类似软件，最初名字叫DKBTrace
。 他把它贴在了一个论坛上面，以为别人会对它感兴趣。 1987, Aaron
Collins下载了DKBTrace然后开始了?[x86](https://baike.baidu.com/item/x86/6150538)机器的移植工作.
他和David
Buck一起合作为它添加了更多的功能特性。直到这个软件更加的流行，他们已经为了加新功能而应付不过来的时候。
1989, David
把这个项目变成了一个程序员团队合作的工程。这时候，他觉得已经没有资格来命名软件名字了。所以考虑了很多新的名字。\"STAR\"
(为动画和渲染而生存的软件：Software Taskforce on Animation and
Rendering) 是一个最初的考虑, 但是最后变成了
\"持续不断更新版本的光线跟踪引擎Persistence of Vision Raytracer,\"
简写为\"POV-Ray\"

## Intel OSPRay（CPU）

OSPRay是[Intel公司]{.mark}在[科学可视化领域]{.mark}目前先进的CPU光线追踪引擎（框架），基于底层的[embree光线追踪]{.mark}框架进行构建的，可以理解为OSPRay是整合了更多内容的更高级的框架，目标是用于科学可视化领域的体渲染或者光线追踪渲染应用。

**实时光线追踪分子可视化**

OSPRay用来开发科学可视化应用可谓是非常方便，因为本身框架很高级别了，连摄像机模型都已经封装好了，甚至渲染器也已经封装好了，开发的时候直接调用就行，只需要根据需求设置参数，不需要写过多的底层代码。因为是基于[embree框架]{.mark}构建的，所以OSPRay也是只需要CPU即可完成渲染，没有GPU也完全没问题。

**离线光线追踪"Embree"**，**集合了Intel自行开发的一系列高性能光线追踪内核**，优化支持SSE、AVX等最新处理器指令集，可进行照片级的渲染，速度也比以往提升了100％，此外它还提供了一个照片级渲染引擎实例。

[Embree]{.mark}使用的是**蒙特卡罗(Monte
Carlo)光线追踪算法**，其中绝大多数光线都是不相关的。

英特尔推出 oneAPI 渲染工具包：拥有光线追踪和渲染功能.

IT之家 8 月 27 日消息 根据英特尔官方的消息，在 SIGGRAPH 2020
会议上，英特尔发布了oneAPI
渲染工具包的最新产品，英特尔表示该渲染工具包可为图形与渲染行业带来顶级的性能和保真度。

## AMD Radeon ProRender

已经在Blender中使用。

## Nvidia IndeX（GPU）

1.  微软的DirectX
    Raytracing（DXR）API。将光线追踪功能完全集成到游戏开发者所采用的行业标准API
    DirectX中，使光线追踪成为光栅化与计算的补充，而非其替代品。DXR专注于通过光栅化和光线追踪技术的混合型技术，来处理用户案例。

2.  NVIDIA的Vulkan光线追踪扩展程序。是Vulkan图形标准的光线追踪扩展程序，也是在跨平台API中实现光线追踪和光栅化技术紧密耦合的另一种途径。

3.  NVIDIA的OptiX
    API。是基于GPU实现高性能光线追踪的应用程序框架。它为加速光线追踪算法提供了一个简单、递归且灵活的管线。OptiX
    SDK包含两个可相互独立使用的主要组件：用于渲染器开发的光线追踪引擎和post
    process管线来处理最终显示的像素。

NVIDIA IndeX is a [3D volumetric interactive visualization SDK]{.mark}
that allows [scientists and researchers]{.mark} to visualize and
interact with massive data sets, make real-time modifications, and
navigate to the most pertinent parts of the data, all in real-time, to
gather better insights faster. IndeX leverages GPU clusters for
scalable, real-time, visualization and computing of multi-valued
volumetric data together with embedded geometry data.

PARAVIEW PLUGIN FOR WORKSTATIONS AND HPC CLUSTERS

There are two versions of the plug-in. For usage in a workstation, or
single server node, the plug-in is available at no cost. For performance
at scale in a GPU-accelerated multi-node system, the Cluster edition of
the plug-in is available at no cost to academic users and with a license
for commercial users.

## NVIDIA Optix

[Ingo Wald]{.mark}

# 科学可视化与计算机图形渲染以及一些程序介绍

近年来，科学数据的规模日益增大，科学可视化技术的发展迅猛，大数据情况分下的科学数据可视化及计算机图形渲染技术的引入，是一个前沿技术。我在github上收集了一些此类程序，主要是使用[ParaView及相关图形渲染开源程序，如Blender，]{.mark}实现的图形可视化库，记录和介绍如下：

## 地球科学数据可视化

[mobigroup]{.mark}开发了很多地球科学数据，使用ParaView可视化的程序，如：MantaFlow,
地形3D线框模型、近海岸环境等.......

pv_atmos: 采用ParaView Python
API可视化大气和海洋的netcdf格式数据的程序。

PVGeo: 主要是地球物理数据的可视化，结构数据。

PyVista: Python VTK

## Blender渲染工具

### ParaView-Omniverse-Connector

ParaViwe
5.10可使用Connector通过Omniverse与其他很多图形渲染软件交互，如Blender，实现数字孪生。

### Blender的VTK插件

地理信息系统的渲染：Blender GIS (https://github.com/domlysz/BlenderGIS)

## 其他

### Paraview-to-POVRay-Water-Render 

将等值面输出，然后使用光追技术将其渲染成自然水面的样子！

### VTK2Blender

BVKTNodes：直接将VTK（结构和非结构）数据读取到Blender进行渲染

<https://bvtknodes.readthedocs.io/en/latest/BVTKNodes.html>

### Vulkan

Vulkan一直以来都是用于动画制作的图形渲染，从2016年开始法国的Rossant将Vulkan引入科学数据可视化研究，开发了Datoviz，但处于起始阶段。

Cyrille Rossant, Nicolas Rougier. High-Performance Interactive
Scientific Visualization With Datoviz via the Vulkan Low-Level GPU API.
Computing in Science and Engineering, Institute of Electrical and
Electronics Engineers, 2021, 23 (4), pp.85-90.
