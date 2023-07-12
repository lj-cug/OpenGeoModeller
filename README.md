# OpenGeoModeller

## 介绍

地球科学模拟系统，组件包含：区域性气象模式、地表水、地下水、地震波正演等数学模式，以及相应的前后处理程序。

## 仓库内容

1.  Meterology: 气象模式，WRF, RegCM
2.  SurfaceWater: 地表水模拟系统，SCHISM, telemac, shyfem, SUNTANS, DGSWE
3.  GroundWater: 地下水及多孔介质流动模拟，包含：MODFLOW, MPLBM-UT
4.	Seismics: 地震数据处理与解释，OpendTect-6.6, Dave's JTK (Java)
5.  VirtualReality: 虚拟现实(VR)，在线可视化(In-situ)
6.  HPC-Base: 高性能计算基座，包含：软硬件安装, OpenMP, MPI, CUDA (GPGPU), OneAPI, HPX等
7.  ESM-Coupler: 地球系统模式+耦合器，包含：RegESM, ESMF, BMI等
8.  Reservoir: 油气藏模拟，黑油模式(OMP),水力压裂(GEOSX, VPFHF),地热模拟waiwera, THMC模拟OpenGeosys
9.  Geophysics: 地球物理正演及反演，OpenSWPC, Seissol, 层析成像TOMO3D, 全波形反演Devito,JUDI
10. ShallowWater: 浅水方程相关的模拟, Titan2d (滑坡), OP2-Volna(海啸-DSL), Triton (GPU-Flood), LISFLOOD-FP-8.0 (DG2-Flood)
11. Hydrology：基于物理机制的分布式水文模拟及其耦合模拟

## 仓库建设目的

1.  开源：摆脱商业软件的制约，解决"卡脖子"问题
2.  高效：Python脚本自动化建模工作流，提高建模效率和可重复
3.  高性能：多并行机制加速模拟，提高数值模拟时空分辨率
4.  系统性：实现不同介质耦合模拟，体现地球科学的系统性
5.  易用性：快速编译安装和运行，用于解决实际工程问题
6.  可操作性：完善的操作流程说明，实现"傻瓜操作"

## 安装教程

1.  参考各组件的文档说明
2.  基于Linux操作系统(Ubuntu 20.04 OS)
3.  基于开源代码

## 合作与共赢

有对这个项目感兴趣的同仁，一块研究，提高地球科学数值模拟的研究。

作者简介：李健，QQ: 94207625        	email: jianli@cug.edu.cn   
		  [微信QR](./QR-code.jpg)
