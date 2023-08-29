# Survery-GeologicalModel

## Principles

测井解释和地震解释的基本原理学习记录

## Opendtect-6.6

地震解释的开源软件的教学课件。OpendTect-6.6在持续开发中，其特色是地震数据解释的并行计算和批处理。OpendTect-6.6使用C++语言开发
目的：在OpendTect-6.6的地震解释工作流的基础上，后期将进一步挖掘地震解释的自动化工作流和并行计算，提高对大规模地震数据解释的工作效率

## Opendtect6.6-Practices

OpendTect软件的实战操作

## GMT

地球科学领域常用的作图的程序，可在Opendtect中使用

## Madagascar

可重复的开源的地震数据处理程序，最新版本 V4.1 

可在Opendtect中，基于GUI使用Madagascar

## JTK (Java ToolKits)

美国科罗拉多矿业学院Dave Hales与中国科学技术大学Wu Xinming教授，使用Java语言开发的地震解释工具

### OSV

断层自动解释

### MHE

层位自动解释

### HPC开发

可使用java-cuda和java-MPI针对地震属性解释中的核函数，开展提高解释效率的开发

## AVO

AVO技术及程序说明


# 地质建模

使用近年来开源的隐式地质建模Python脚本工具，实现地质建模（即多孔介质流动模型的前处理）

基于地震勘探的地震数据解释得到的断层和层位，执行隐式地质建模

## Implicit-GeoModeling (隐式地质建模与显式地质建模)

目前多数的地质建模软件，如Petrel和SKUA-GOCAD，都是使用显式地质建模，也就是建模过程必须显式地构建出地质体，如断层，需要大量的手工工作量

而隐式地质建模，仅需要地质体的梯度或dip-azimuth等地震属性数据的势场插值，就能得到地质建模文件，无需手工介入，但由于对计算机内存和计算性能要求较高，近5年来才有可用的计算机程序供使用

当前的隐式建模Python工具主要是：GemPy和LoopStructural

## Gempy

德国亚琛工业大学开发的隐式地质建模Python程序

## LoopStructural

澳大利亚开发的隐式地质建模Python程序

## CPM

角点网格的数据结构及格式转换