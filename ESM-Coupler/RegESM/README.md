# RegESM

区域性地球耦合系统
包括：RegCM/WRF + ROMS/MITgcm + WAM (Wave Model) + RTM (Hydrological River Model) + COP (Co-processing by ParaView)

ESMF耦合器+ ParaView Offscreening + In-situ visualization(原位可视化)

## 本仓库文档

- [install](./install/)：GTC-2018-demo 等安装脚本
- [doc](./doc/)：部署指南、版本说明与问题汇总

## install

安装GTC-2018-demo的bash脚本程序，见 [install](./install/) 与 [install/README.md](./install/README.md)

## 参考文献

Ufuk UtkuTuruncoglu, Sylvia Murphy, Cecelia DeLuca, Nuzhet Dalfes. 2011. A scientificworkflowenvironmentforEarthsystemrelatedstudies. Computers &Geosciences, 37: 943–952

RegESM (Regional Earth System Model) User Guide. Version: 1.0, May 2014. U. U. Turuncoglu, Istanbul Technical University, Informatics Institute, Turkey. 

U. U. Turuncoglu, G. Giuliani, N. Elguindi, and F. Giorgi. 2013. Modelling the Caspian Sea and its catchment area using a coupled regional atmosphere-ocean model (RegCM4-ROMS): model design and preliminary results. Geosci. Model Dev., 6.

Ufuk Utku Turuncoglu, Gianmaria Sannino. 2017. Validation of newly designed regional earth system model (RegESM) for Mediterranean Basin. Clim Dyn, 48: 2919–2947.

Ufuk Utku Turuncoglu. 2019. Toward modular in situ visualization in Earth system models: the regional modeling system RegESM 1.1. Geosci. Model Dev., 12, 233–259.

L. E. Sitz, F. Di Sante, R. Farneti, e al. 2017. Description and evaluation of the Earth System Regional Climate Model (Reg CM-ES). J. Adv. Model. Earth Syst., 9,
doi:10.1002/2017MS000933.

## 相关文档

- [ESM-Coupler](../)：学科入口
- [ESMF](../ESMF/)：耦合框架
- [Meteorology/WRFV4](../../Meteorology/WRFV4/)：大气分量
- [SurfaceWater/ROMS](../../SurfaceWater/ROMS/)：海洋分量
- [In-situ-Visualization](../../VirtualReality/In-situ-Visualization/)：原位可视化
- [hpc-base](../../hpc-base/)：编译与并行环境
