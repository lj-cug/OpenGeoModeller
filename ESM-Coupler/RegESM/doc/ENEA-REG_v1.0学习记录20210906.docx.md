# Med-CORDEX项目的新建立的ENEA-REG (RegESM)耦合模拟系统

Alessandro Anav, Adriana Carillo, Massimiliano Palma, Maria Vittoria
Struglia, Ufuk Utku Turuncoglu, and Gianmaria Sannino. The ENEA-REG
system (v1.0), a multi-component regional Earth system model:
sensitivity to different atmospheric components over the Med-CORDEX
(Coordinated Regional Climate Downscaling Experiment) region. Geosci.
Model Dev., 14, 4159-4185, 2021

摘要：建立新的应用于Med-CORDEX (Coordinated Regional Climate Downscaling
Experiment)地区的区域性ESM---ENEA-REG。

ENEA-REG的气象模式有2个：RegCM和WRF，还有HD，一个海洋模式MITgcm，不同复杂度的陆地表面模型。模型组件由ESMF耦合，驱动器RegESM。

模型模拟时段：1980\~2013，采用ERA-Interim再分析数据驱动。结果表明：2个大气模式都能复演欧洲地中海地区的大尺度和局部特征，但还是发现一些差异：[WRF模式在计算域的东北边界附近的冬季有明显的冷偏差，在整个欧洲大陆区域的夏季有暖偏差]{.mark}；[而RegCM模式过高估计了整个地中海地区的风速]{.mark}。

类似地，海洋模式能正确复演海洋特征。

# 2模型介绍

## 2.1RegESM耦合器

![](./media/image1.emf){width="4.108417541557305in"
height="4.112110673665792in"}

图1 ENEA-REG区域耦合模型的组件示意图

## 2.2大气模式组件：WRF和RegCM

WRF模式V3.8.1_cpl，ARW；RegCM_v4.5 (Giorgi et al.,
2012)：[尚未实施非静水压力动力核心]{.mark}。2个大气模式使用的主要物理参数化方案如下：

![](./media/image2.emf){width="5.065878171478565in"
height="2.9801181102362206in"}

## 2.3海洋模式组件：MITgcm

MITgcm_c65 (Marshall et al., 1997)

## 2.4河流径流模型：HD

HD_v1.0.2，使用矩形全球网格，水平分辨率0.5°，受每日地表径流和集水数据驱动。HD模型代码做了小的修改(Turuncoglu
and Sannino,
2017)：从大气模型获取地表径流和集水，向海洋模式提供河道流量。

# 3试验设计和观测数据集

模型[校核(validation)]{.mark}时间段：1982\~2013，使用前2年的spin-up模拟，初始化耦合系统中不同组件的初始场。耦合模型主要校核的场有：海洋模式的海表温度、海表盐度和海洋的混合层深度，以及大气模式的2m气温、风速、淡水和热通量。HD：比较Po
River的河道径流模拟结果。

[校核：]{.mark}

（1）模拟的SST结果与NOAA研发和发布的数据比较Objectively Interpolated Sea
Surface Temperatures (OISST v2, Reynolds et al., 2002 and 2007), The
OISST composites observations from different platforms (satellites,
ships, buoys) on a 1=4°global grid and the gaps are filled by
interpolation (Reynolds et al., 2007).

（2）盐度数据：地中海的MEDHYMAP (Jordà et al., 2017). For the mixed
layer depth, we use a global climatology computed from more than 1
million Argo profiles collected from 2000 to the present (Holte et al.,
2017); this climatology provides estimates of monthly mixed layer depth
on a global 1°gridded map.

（3）大气模式的校核，采用ERA5，RegESM能复演ERA5数据(Mooney et al.,
2013)，ERA5提供陆地和海洋的数据信息。ERA5数据对海洋有一些缺陷，用来验证风速时应谨慎使用(Belmonte
Rivas and Stoffelen, 2019)。

（4）观测的河道径流验证HD计算结果。

# 4结果

## 4.1大气模型的评价

在参考期(1982\~2013)的冬季(DJF)和夏季(JJA)，比较RegESM模拟结果与ERA5。

比较：[空间分布形态和异常分布图]{.mark}，还计算[不确定相关模式]{.mark}与[相对实测值的计算域内平均的偏差]{.mark}，来评估模型精度。

如图3，[地表气温]{.mark}的空间分布及异常分布，在冬季与ERA5数据趋势一致，

如图4，降雨量的空间分布及异常分布，

如图5，淡水通量模块（包括：降雨P、蒸发E和E-P）的年内变化和季节平均值，

如图6，地表上方10 m风速的的空间分布及异常分布，

如图7，净热量通量的空间分布及异常分布，

如图8，净热量通量的年内分布及季节平均值，

## 4.2海洋模式的评价

### 4.2.1海表过程 {#海表过程 .标题3}

海表温度（SST）和海表盐度（SSS）

如图9，SST的空间分布及与OISST观测值的偏差，

如图10，海表温度的月平均值异常和季节平均值异常的计算与观测对比。

如图11，海表盐度的空间分布及与MEDHYMAP观测值的偏差，

如图12，Po河流的径流的季节平均值与RivDIS观测数据集的对比，

如图13，海表盐度的月平均值异常和季节平均值异常的计算与观测对比。

### 4.2.2海表高度及循环 {#海表高度及循环 .标题3}

如图14，WRF-MITgcm与RegCM-MITgcm分别模拟的海面高度与水下30m处水动力场

### 4.2.3热量和盐度 {#热量和盐度 .标题3}

### 4.2.4深水形态 {#深水形态 .标题3}

如图18，

# 5结论

2个大气模式(WRF与RegCM)都能复演大尺度或地中海局部的大气特征。但有偏差，。。。

海洋模式中，发现在冬季月份的偏冷预估的SST，是由于RegCM过高计算了风速。

降低上述偏差的可行方法就是采用[插值(nudging)技术减小偏差(Liu et al.,
2012)。]{.mark}该方法就是：不仅传递驱动信息给侧向边界，还传递给区域计算域内部(Heikkilä
et al.,
2011)，这样保证了GCM气象模式与RCM的一致性，但目前仍[存在一些争议(Omrani
et al., 2015)]{.mark}：

一方面，插值技术不允许RCM太多偏离驱动场，这限制了RCM的内部物理机制(Sevault
et al., 2014; Giorgi, 2019)。Sevault et al.,
2014指出nudging强烈地限制了大气流动的天气模式(synoptic
chronology)，因此也限制了海气通量的表现和海洋的反馈，他们还发现：这便于逐日和年内的与观测数据的比较，但nudging也限制了耦合模型中大气模式组件的内部变化。

相反地，关于地中海的极端事件的耦合大气海洋模型的研究表明Lebeaupin-Brossier
et al.
(2015)：Nudging没有抑制小尺度过程，因此正确地模拟了潜在的海气反馈。该结论与Omrani
et al.,
2015的一致，建议nudging技术不会影响小尺度场，因为仅松弛了大尺度过程(only
the large scales are relaxed)。

RegCM模式没有nudging功能，因此使用WRF模式中的谱插值技术应用于RegESM
([WRF-MITgcm]{.mark})。计算结果表明（[如图20]{.mark}），采用插值技术后提高了气象变量的计算精度，有趣的是：海洋模式的物理过程模拟精度也提高了（与观测数据比较后），特别是中部和深度区域的形态。

# 参考文献

Skamarock, W. C. and Klemp, J. B.: A time-split nonhydrostatic
atmospheric model for weather research and forecasting applications, J.
Comput. Phys., 227, 3465--3485, 2008. [WRF模式]{.mark}

数据同化的作用：

Liu, P., Tsimpidi, A. P., Hu, Y., Stone, B., Russell, A. G., and Nenes,
A.: Differences between downscaling with spectral and grid nudging using
WRF, Atmos. Chem. Phys., 12, 3601--3610,
https://doi.org/10.5194/acp-12-3601-2012, 2012.

Omrani, H., Drobinski, P., and Dubos, T.: Using nudging to improve
global-regional dynamic consistency in limited-area climate modeling:
What should we nudge?, Clim. Dynam., 44, 1627--1644, 2015.

Giorgi, F.: Thirty years of regional climate modeling: where are we and
where are we going next?, J. Geophys. Res.-Atmos., 124, 5696--5723,
2019.

Sevault, F., Somot, S., Alias, A., Dubois, C., Lebeaupin-Brossier, C.,
Nabat, P., Adloff, F., Déqué, M., and Decharme, B.: A fully coupled
Mediterranean regional climate system model: design and evaluation of
the ocean component for the 1980-2012 period, Tellus A, 66, 23967,
https://doi.org/10.3402/tellusa.v66.23967, 2014.

[RegESM：]{.mark}

Sitz, L., Di Sante, F., Farneti, R., Fuentes-Franco, R., Coppola, E.,
Mariotti, L., Reale, M., Sannino, G., Barreiro, M., and Nogherotto, R.:
Description and evaluation of the Earth System Regional Climate Model
(RegCM-ES), J. Adv. in Model. Earth Sy., 9, 1863--1886, 2017.
