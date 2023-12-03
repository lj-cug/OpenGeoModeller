# WindFarm (风力发电和水轮机发电)

风力发电牧场，评估发电效率、涡流、发电机组和墩子受力分析等，近年来，采用大涡模拟(LES)结合气象模型来耦合模拟分析, 通过文献调研，罗列一些开源模型如下：

风力发电机与水下水轮机有相似之处, 均罗列于此.

## PCUILES

最早的可用于学习LES方法的FORTRAN程序(美国Stanford大学开发).

## Hydro3D

用于海洋水下水轮机发电牧场优化的LES程序, 如何使用和开发, 请参考[Wiki](https://github.com/OuroPablo/Hydro3D/wiki)

## VFS-Rivers & VFS-Geophysics

美国Minnisota大学开发的用于模拟自由表面流动的LES模型, 
明渠河流与泥沙输移的耦合模型VFS-Geophysics (Ali, et al., 2023). 

特点是：贴体曲线网格和PESTc库，可适应弯曲边界, 多种求解器求解Poisson方程.

## WInc3D

基于帝国理工学院开发的Incompact3D-LES和actuator line model (ALM)的风力发电牧场的模型.

Incompact3D是英国帝国理工学院开发的基于紧致差分格式的DNS模型.

## PALM-WTM

德国研发的基于LES的PALM气象模式的风力发电效率评估模型

[PALM-WTM的介绍](https://palm.muk.uni-hannover.de/trac/wiki/doc/tec/wtm#no1)

## WRF_v3.7.1-WindFarm

最早的丹麦研发的基于WRF模式的风力发电效率评价模型, 研究了尾流脱落对发电效率的影响.

Xiaoli G. Larsén and Jana Fischereit. 2021. A case study of wind farm effects using two wake parameterizations in theWeather Research and Forecasting (WRF) model (V3.7.1) in the presence of low-level jets.Geosci. Model Dev., 14, 3141–3158.

## WRF-SADLES

挪威研发的基于WRF模式的风力发电机组的发电效率评估模型, 并于PALM-WTM模型的模拟结果做了对比.

WRF-SADLES的特点是：使用简化的Simple Actuator Disc模型，需要的信息较少

Hai Bui, et al. Implementation of a Simple Actuator Disc for Large Eddy Simulation (SADLES-V1.0) in the Weather Research and Forecasting Model (V4.3.1) for Wind Turbine Wake Simulation. https://doi.org/10.5194/egusphere-2023-491
