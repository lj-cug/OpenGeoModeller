# General Lake Model (GLM)

湖泊, 属于内陆浅水区域, 通常伴随气象驱动下的内部流体动力过程(主要为翻转)和水面的热力学过程(热交换+蒸散发),
以及生物地球化学过程(biogeochemical process).

目前, 针对湖泊水体开发的1D浅水水动力及生态模拟有：
```
1 PCLake + FABM + GOTM (Hu, et al., 2016)
2 GOTM-lake (https://github.com/gotm-model/code)
3 General Lake Model (GLM) + FABM + AED2 (Hipsey, et al., 2019)
```

## GLM研究计划

(1) GLM模型的垂向涡扩散系数计算模块功能欠缺, 目前仅有1个参数化的计算方法, 考虑调用GOTM的turbulence模块, 由于GLM是C语言，需要调用[C接口的GOTM-5.2.1](http://basilisk.fr/src/gotm)

(2) 分布式水文模型SHUD-lake与GLM模型的耦合.

(3) 湖泊模型的数据同化功能, 参考EAT-1D框架(Jorn Bruggeman, et al., 2023)和SCHISM-3D系统(Yu et al., 2022).

## 增强的FABM生态框架: FABM-NflexPD 1.0


## 湖泊模拟的R工具

湖泊系综模拟工具LakeEnsemblR

GLM, GOTM都有前后处理的R工具

## 参考文献

Robert Ladwig, et al. 2021. Lake thermal structure drives interannual variability in summer anoxia dynamics in a eutrophic lake over 37 years. Hydrol. Earth Syst. Sci., 25: 1009–1032.

Lele Shu, et al. 2023. Advancing Understanding of Lake-Watershed Hydrology Through A Fully Coupled Numerical Model. https://doi.org/10.5194/hess-2023-166

Fenjuan Hu, et al. 2016. FABM-PCLake – linking aquatic ecology with hydrodynamics. Geosci. Model Dev., 9, 2271–2278.

Matthew R. Hipsey, et al. 2019. A General Lake Model (GLM 3.0) for linking with high-frequency sensor data from the Global Lake Ecological Observatory Network (GLEON).Geosci. Model Dev., 12, 473–523.

Jorn Bruggeman, et al. 2023. EAT v0.9.6: a 1D testbed for physical-biogeochemical data assimilation in natural waters. Geoscientific Model Development. Discussions. https://doi.org/10.5194/gmd-2023-238

Yu Hao-Cheng, et al. 2022. Development of a flexible data assimilation method in a 3D unstructured-grid ocean model under Earth System Modeling Framework. EGUsphere. https://doi.org/10.5194/egusphere-2022-114

Onur Kerimoglu, Prima Anugerahanti, Sherwood Lan Smith. 2021. FABM-NflexPD 1.0: assessing an instantaneous acclimation approach for modeling phytoplankton growth. Geosci. Model Dev., 14, 6025–6047.

Tadhg N. Moore, et al. 2021. LakeEnsemblR: An R package that facilitates ensemble modelling of lakes. Environmental Modelling and Software. 143: 105101
