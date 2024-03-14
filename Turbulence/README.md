# OpenFOAM (OpenCFD)

一个家喻户晓且不断发展和得到广泛应用的CFD开源软件.
在OpenFOAM框架的基础上开发了很多特殊用途的CFD应用,包括：

1.  基于PDE约束和伴随方法的优化
2.  气动声学
3.  多相流模型
4.  分子动力学模型
5.  精细湍流模型(LES, DNS)
6.  水轮机和风力发电机模拟

## DAFOAM

基于CFD和离散伴随方法的体型优化工具，类似的工具还有：ADFLOW, SU2等

注： ADFLOW使用可压缩的RANS求解器和多块结构网格，适用于机翼体型的气动优化;
SU2是斯坦福大学开发的C++程序, 使用非结构网格, 可快速实施气动优化的整个工作流;
DAFOAM的CFD求解器是OpenFOAM, 其功能性应该是最强的,且使用Python语言,易于使用.

## libAcoustics

俄罗斯人开发的基于OpenFoam框架的气动声学模型

[**UniCFD Web-laboratory**](https://github.com/unicfdlab) was established in ISP RAS (www.ispras.ru) 
in 2011 to develop and spread scientific open source software in Russia.

## ShallowFOAM

浅水方程求解器

## demFoam

集成LAMMPS和OpenFOAM的DEM模型

## sediFOAM

水沙两相流

## mdFOAM

分子动力学

## fastFlume

基于OpenFOAM框架开发的水轮机的模拟程序.

# Turbulence Engineering

侧重湍流细微结构的CFD模型，同时考虑其工程应用价值.

## SU2

斯坦福大学研发的用于机翼体型优化的CFD求解器, 可在Windows, Linux和MacOS系统下使用.

## SimVascular和lifex-CFD

心血管血液流动的全流程CFD模型. 

lifex-CFD是基于deal.II库开发的, 功能强大.

## finesed3d

直接数值模拟(DNS)的水沙两相流模型.

## HiFiLES

基于非结构网格的高阶LES模型, 可运行在GPU集群上. 6年前已经停止研发, 研发人员已经转向FR模型开发团队.

Manuel R. L′opez-Morales et al., 2014. Verification and Validation of HiFiLES: 
a High-Order LES unstructured solver on multi-GPU platforms. 
AIAA Aviation, 32nd AIAA Applied Aerodynamics Conference

# tudflow3d

基于结构网格，用于研究河道疏浚的LES模型，包括水动力和泥沙输移两部分.