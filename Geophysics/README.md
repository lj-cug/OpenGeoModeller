# Geophysics 

地球物理的模拟程序，分为正演和反演模型，数值方法主要采用：有限差分法和有限单元法

# first-arrival-time-pick

4个Python语言编写的初至时间拾取的GUI程序,
初至拾取文件是使用pyGIMLi执行层析反演的输入文件

# pyGIMLi

pyGIMLi是一款高效的使用Python语言编写的地球物理反演工作,
地震层析反演，可作为FWI的初始速度模型

## OpenSWPC

交错结构网格模式下，使用**有限差分法**的地震波传播正演模型

## defmod-Openswpc

地壳变形模拟defmod与地震波传播的有限差分模型OpenSWPC的耦合模拟，研究地震波生成-传播的整体过程

## Seissol

非结构网格模式下，使用**ADER-DG高阶间断Galerkin有限单元法**的地震波传播正演模型

## Devito + JUDI

基于**有限差分特定域语言Devito库**，以及Julia语言开发的JUDI库，快速实现全波形反演(FWI)全过程

## spyro + Firedrake

采用**间断Galerkin法和波形自适应非结构网格技术**的全波形反演模拟，特点是在非结构网格上求解波动方程
