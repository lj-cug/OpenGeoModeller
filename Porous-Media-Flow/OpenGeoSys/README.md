# OpenGeoSys6

Thermo-hydro-mechanical/chemical (THMC)模型, 由德国亥姆霍兹研究所研发

OpenGeoSys-6开始于2011年，2012年开源(4-cluase BSD license), 最新版本6.4.4

OpenGeoSys-6基于OpenGeoSys-5 (Kolditz et al., 2012), 直到2015年发布OpenGeoSys-6.0.1,
在2016年实施了基于PETSc的并行计算框架, 2017年实施了很多物理过程框架(production-ready).

有限单元法

编程语言：C++

前后处理Python工具： [ogs5py](https://github.com/GeoStat-Framework/ogs5py), 
[ogs6py](https://github.com/joergbuchwald/ogs6py), OpenGeoSys Data Explorer

[ogs5py_benchmarks](https://github.com/GeoStat-Examples/ogs5py_benchmarks)

之前大量的用户和开发者都是基于ogs5, 转向ogs6会有一些困难

## ogs6示例

[ogs6-tutorial只有一个沉积盆地冰川移动](https://www.opengeosys.org/docs/tutorials/advancing-glacier/)

Complete workflow for simulating a geological model with time and space dependent boundary conditions (advancing glacier)

[ogs6-UserGuide](https://www.opengeosys.org/docs/userguide/basics/introduction/)

[ogs6-benchmarks](https://www.opengeosys.org/docs/benchmarks/)

[THMC-benchmark](https://www.opengeosys.org/docs/benchmarks/thermo-hydro-mechanics/thermohydromechanics-bgr/)

[Running OGS in a container](https://www.opengeosys.org/docs/userguide/basics/container/)
Ubuntu OS: apt install singularity

[Running OGS with MPI](https://www.opengeosys.org/docs/userguide/features/parallel_computing_mpi/)

## 参考文献

O. Kolditz, et al. 2012. OpenGeoSys: an open-source initiative for numerical simulation of thermo-hydro-mechanical/chemical (THM/C) processes in porous media. Environ Earth Sci,67:589–599, DOI 10.1007/s12665-012-1546-x

Lars Bilke, et al. 2019. Development of Open-Source Porous Media Simulators: Principles and Experiences. Transport in Porous Media, 130: 337–361

Karsten Rink. OpenGeoSys Data Explorer Manual. 2023
