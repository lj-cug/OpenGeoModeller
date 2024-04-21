# RegCM_v4.7手册学习

# 第1章 RegCM

美国气象学会对RCM的定义（https://glossary.ametsoc.org/wiki）：

A regional climate model (Abbreviated RCM) is a numerical climate
prediction model forced by specified lateral and ocean conditions from a
general circulation model (GCM) or observation-based dataset
(reanalysis) that simulates atmospheric and land surface processes,
while accounting for high-resolution topographical data, land-sea
contrasts, surface characteristics, and other components of the
Earth-system. Since RCMs only cover a limited domain, the values at
their boundaries must be specified explicitly, referred to as boundary
conditions, by the results from a coarser GCM or reanalysis; RCMs are
initialized with the initial conditions and driven along its
lateral-atmospheric-boundaries and lower-surface boundaries with
time-variable conditions. RCMs thus downscale global reanalysis or GCM
runs to simulate climate variability with regional refinements. It
should be noted that solutions from the RCM may be inconsistent with
those from the global model, which could be problematic in some
applications.

RegCM是区域性气候模型(RCM)，由Filippo
Giorgi开发，经过几十年的发展。第1代RegCM开发于1980s Dickinson et al.
(1989)和Giorgi (1990)，to later versions in the early nineties (RegCM2,
Giorgi et al. \[1993b\], Giorgi et al. \[1993c\]), late nineties
(RegCM2.5, Giorgi and Mearns \[1999\]), 2000s (RegCM3, Pal et al.
\[2000\]) and 2010s, (RegCM4, Giorgi et al. \[2012\])

RegCM研究范围从区域性气候模拟研究，到过程研究，到古气候和成熟的未来气候预测研究(Giorgi
and Mearns \[1999\], Giorgi et al. \[2006\], Giorgi \[2014\])。

RegCM是一个社区模型，主要用于一些工业化国家和发展中国家的气候研究(Pal et
al., 2007)。其目标是促进发展中国家的气候研究。得到了ICTP的Regional
Climate research NETwork (RegCNET)的支持。

# 第2章 介绍

## 2.1RegCM的历史

[区域性大气模拟]{.mark}是由Dickinson et al. (1989)和Giorgi
(1990)提出的，基本思路就是：基于[one-way
couplin]{.mark}的思路，大尺度的General Circulation Model
(GCM)初始的和与时间有关的气象边界条件(lateral
BC)，支持高分辨率的Regional Climate Model
(RCM)，但没有来自RCM的反馈来驱动GCM。

### 第1代RegCM

[第1代NCAR RegCM]{.mark}是基于NCAR-Pennsylvania State University (PSU)
Mesoscale Model version 4 (MM4) in the late 1980s (Dickinson et al.,
1989; Giorgi, 1989).
RegCM_v1模型的动力学核心是源于MM4，是可压缩的有限差分模型，采用静水压力平衡和垂向σ坐标。之后，使用分裂的显格式时间积分格式，减小在大的地形梯度区域的水平扩散效应(Giorgi
et al., 1993a, b)。因此，RegCM的动力学核心类似于静水压力版本的MM5(Grell
et al.,1994)。

RegCM_v4的动力学核心因此是可压缩的，sigma-p垂向坐标，Arakawa-B网格，其中风和热力学变量在水平向是交错分布，使用分裂的显格式时间积分，2个快速的重力波模式独立求解，然后使用较小的时间步长积分计算。

RegCM_v1使用BATS (Biosphere-Atmosphere Transfer
Scheme)的陆地表面模型、辐射传输格式使用CCM_v1 (Community Climate Model
version
1)、中等分辨率的局地行星边界层模型、Kuo-type的积云对流格式和显格式的湿度模型(Hsie
et al., 1984)。

### 第2代RegCM

第一次RegCM的物理和数值格式的升级记录见Giorgi et al., 1993a,
b，产生第2代RegCM ([RegCM_v2]{.mark})。RegCM_v2的物理模块是基于NCAR
Community Climate Model version 2 (CCM2)和中尺度模式MM5 (Grell et al.,
1994)。

使用CCM2的辐射传输程序包(Briegleb,
1992)计算辐射，非局部的行星边界层模型代替了老的局部模型、增加了一个质量通量的积云模型(Grell,
1993)，还增加了最新版本的BATS1E (Dickinson et al., 1993)。

### 第3代RegCM

第3代RegCM增加了很多参数化模块，主要是参考了最新的CCM3 (Kiehl et al.,
1996)。

CCM3的辐射传播模型代替了老的辐射模型，考虑了H~2~O, O~3~, O~2~,
CO~2~和云的影响。太阳辐射传输采用δ-Eddington方法处理，云辐射依赖于3个云参数：云覆盖比、云中液态水含量和云中的有效水滴半径。

> CCM3保持了CCM2的结构，但增加了新的特性，诸如：温室气体(NO~2~, CH~4~,
> CFC)的影响、大气气溶胶、云冰。基于气溶胶光学特性，考虑了气溶胶引起太阳辐射的散射和吸收。

增加了简化的显式湿度模型(Hsie et al.,
1984)，仅使用云水的诊断方程，考虑云水形成、对流和紊流掺混、欠饱和情况下的再蒸发、转化为雨。云辐射计算中直接使用预测的云水变量，而不使用局部的相对湿度；增加了一个很重要的模拟的水文循环和能量收支之间的相互作用的模块。

太阳光谱特性是基于云水路径，。。。

另外，计算的云冰百分比是温度的函数。。。

### 第4代RegCM

RegCM_v4修改了[辐射云格式]{.mark}：在网格节点处计算云的总覆盖，然后单独地对云层和清空部分计算表面通量。

模型网格单元中的总云层覆盖取一个中间值。。。

一个大尺度的云和降雨模型，考虑云的压格子尺度变化(Pal et al.,
2000)，海表面通量的参数化(Zeng et al., 1998)和多个积云对流模型(Grell,
1993; Emanuel, 1991; Emanuel and Zivkovic-Rothman,
1999)，这与RegCM_v3一样，但RegCM_v4引入一个新的[\"混合格式\"
Grell+Emanuel]{.mark}，这允许用户选择使用2个格式中的一个，是海洋-陆地掩膜的函数。

[气溶胶辐射传输计算]{.mark}是RegCM_v4的新发展，增加了红外光谱计算(Solmon
et al., 2008)。RegCM_v3中仅考虑短波光谱的散射和吸收的气溶胶模型(Giorgi
et al.,
2002)。这对于大范围的灰尘(Dust)和海盐颗粒非常重要，通过引入一个气溶胶红外辐射来考虑，红外辐射是气溶胶路径、计算自气溶胶尺寸分布的吸收断面和长波折射率的函数。[长波扩散]{.mark}与大的灰尘颗粒尺寸有关，但模型中没有考虑。

地形和土地利用的压格子尺度空间各向异性使用mosaic-type的参数化方案(Giorgi
et al., 2003b)，允许在BATS1E模型中使用更高的地表分辨率。

RegCM_v4在静水压力动力学核心的基础上，增加了[非静水压力模块]{.mark}，还有微物理选项、多个经典的对流和边界层模型以及一个与Community
Land Surface Model，CLM_V4.5的陆地表面模型的接口和气相化学模块。

[最新版本的RegCM]{.mark}还增加了：在Arakawa
C网格上的基于H垂向坐标的MOLOCH的非静水压力动力学核心、隐格式的Euler向后时间积分格式来计算声波传播、二阶向前-向后格式计算水平向动量方程的时间积分、TVD的对流项计算允许更大的时间步长、用户可选时间步长计算物理参数化方案的贡献。

## 2.2模型组成

[RegCM]{.mark}由4部分组成：Terrain(地形处理),
ICBC（初始条件和边界条件处理）,
RegCM（计算主程序），和Postprocessor（前处理程序）。

Terrain, ICBC是[RegCM]{.mark}前处理的2部分。地表数据（包括elevation,
landuse,
SST）和3D等压(isobaric)气象数据在水平上由经纬度网格插值到一个高分辨率的计算域上，可选择使用Normal
or Rotated Mercator, Lambert Conformal, or Polar Stereographic
Projection等投影方法。

还要实施从GCM分层垂向插值到RegCM的垂向坐标系。

## 2.3RegCM模型的水平网格

有限差分模型，与交错网格关系密切，计算模型方程中的梯度和平均项。

### 2.3.1 MM5水平向Arakawa-B网格

Arakawa-Lamb B交错网格（速度与标量交错存储），如图2.1，其中，标量（T, Q,
p等）在网格单元中心定义，而流速U和V位于节点上。

输入到模型的数据，前处理程序将做必要的[插值]{.mark}，保证网格的一致性。

![](./media/image1.emf)

图2.1 水平Arakawa B交错网格示意图：点(U, V)和叉（T, Q,...）

### 2.3.2水平向Arakawa C网格

如图2.2，标量（T, Q,
p等）定义在网格单元中心，而速度（U，V）定义在网格节点，但不在同一个位置。同样前处理程序对输入数据做插值。

![](./media/image2.emf)

图2.2水平Arakawa-C交错网格示意图：绿点为U，蓝点为V，叉点为标量

## 2.4RegCM模型垂向网格

所有状态变量都定义在各垂向分层的中间处，即为[半分层]{.mark}(half-level)，如图2.3中的虚线。垂向速度定义在[全分层(full-level)]{.mark}处（实线）。

定义sigma分层时，列出全分层，包括在σ=0和σ=1。因此，模型分层总是比完全sigma分层数少1.

从压力分层插值到σ坐标系统。接近地表的σ表层跟踪地形（σ=1），高层σ近似顶部等压面（σ=0）。

![](./media/image3.emf)

图2.3
基于压力分层的模型垂向结构示意图：图示为KZ垂向分层，虚线表示全sigma分层，实现表示半sigma分层。

### 2.4.1基于压力的垂向坐标

MM5动力学核心的垂向坐标是地形跟踪的[压力]{.mark}，见图2.3，意思是：底层网格跟随地形，而最顶部表面是平整的，使用用户施加的刚盖假设的压力。中间层向顶部分层的压力逐渐较小。

静水压力求解器使用无量纲的σ坐标，根据模型分层处的压力p定义

如图2.3，σ坐标在顶部为0，地表处为1，模型各分层用σ值定义。

### 2.4.2基于H的垂向坐标 {#基于h的垂向坐标 .标题3}

MOLOCH动力学核心使用基于H的地形跟踪坐标，如图2.4，是将Z垂向间距\[h,
Z~top~\]转换为规则间距![](./media/image4.emf)：

![](./media/image5.emf)

图2.4基于H的模型分层结构示意图

## 2.5地图投影和地图尺度因子

RegCM有4种投影方式：中纬度地区适合使用Lambert
Conformal投影。低纬度地区适合使用Normal Mercator投影。

[地图尺度因子m]{.mark}= 模型网格距离/实际的地球距离

m在模型方程中不断出现。

# 第3章 模型方程

RegCM模型求解一套描述大气运动的动力学方程，使用多个物理过程的参数化格式：

-   Radiation (Short Wave and Long Wave)

-   Convection

-   Turbulent Diffusion

-   Moist (Clouds and Precipitation)

-   Fluxes exchange with surface (Soil model and Ocean fluxes)

-   Tracer transport and chemistry (Aerosols and full chemistry)

动力学方程在3D计算网格上使用有限差分格式离散，3D网格使用固定的水平分辨率和地形跟踪的垂向坐标。

## 3.1动力学

RegCM有3个动力学核心：

-   静水压力方程求解器

-   使用压力坐标的非静水压力方程求解器

-   使用高度坐标的非静水压力方程求解器

3个求解器的原始方程是不同的，使用不同的预测变量来反映大气状态。

### 3.1.1静水压力动力学核心

（1）控制方程

静水压力动力学方程和数值离散方法见Grell et al. (1994)---MM5 Manual。

垂向![](./media/image6.wmf)坐标用压力定义：![](./media/image7.wmf)

![](./media/image8.wmf)

水平动量方程：

![](./media/image9.wmf)

![](./media/image10.wmf)

连续方程(Sigma_dot方程)

热动力学方程(Omega方程)

静水压力方程：用虚拟温度*T~v~*计算地转势高度

（2）数值方法

水平对流方程离散：统一的标量变量的对流方程，使用(3.14)项计算总的时间步场的变量变化趋势(tendency)。

[参考Grell (1994)报告的第10页：]{.mark}

对流的静水压力（空间）有限差分。

静水压力和非静水压力模型的时间差分使用带Asselin过滤的蛙跳格式。应用于所有变量：

![](./media/image11.wmf)

式中，![](./media/image12.wmf)是过滤后的变量。对所有变量，系数![](./media/image13.wmf)取值0.1。

为了稳定性，所有变量的扩散项都在t-1时层计算，湿度物理过程的相关项计算也一样。

时间分裂：静水压力和非静水压力核心程序都使用时间分裂法。

[参考Grell (1994)报告]{.mark}

[数值计算实施参考图3.1]{.mark}，假设需要在交叉点G上执行(update?)计算，在点a,
b, c, d上执行平均计算。对u/m和v/m项，分别使用点AC, BD, CD,
AB上的变量值计算其平均值。对*p*\**X*项，模型使用gold+cyan和gold+green叉点的值，做加权计算得到加权平均值，增加上游点对局部Courant数的最大设置因子函数的贡献：

![](./media/image14.emf)

其中，f1和f2是1D对流方程计算的局部Courant数，乘以一个控制因子：

![](./media/image15.emf)

dt是模型时间步长(s)；dx,
dy是网格大小(m)，![](./media/image16.wmf)是namelist中设置的参数。在namelist中可将*f~1~*和*f~2~*设为0，关闭水平向对流的迎风模式。

风速分量的水平对流项，需要对水平动量对流项做3次平均。

![](./media/image17.emf)

图3.1 格式描述，显示水平向的对流计算交错格式：圆圈是U,V点，X是标量变量点

消除快速变化重力波的[分裂显格式时间步]{.mark}

垂向模式的初始化

### 3.1.2使用压力坐标的非静水压力方程求解器

非静水压力模型的动力学方程和数值离散方法描述见Grell et al.
\[1994\]---MM5

模型方程(Dudhia, 1993)如下：

水平向动量方程：

声波（即时间积分）。

### 参考文献

Georg A. Grell, Jimy Dudhia, David R. Stauffer. 1994. A Description of
the Fifth-Generation Penn State/NCAR Mesoscale Model (MM5).
NCAR/TN-398 + STR, NCAR TECHNICAL NOTE

### 3.1.3使用高度坐标的非静水压力方程求解器 {#使用高度坐标的非静水压力方程求解器 .标题3}

求解垂向速度的隐格式

## 3.2物理参数化

### 3.2.1辐射格式(Radiation Scheme)

基于NCAR CCM3 (Kiehl et al., 1996)，简言之就是太阳辐射模块，考虑O~3~,
H~2~O, CO~2~, 和O~2~的影响采用δ-Eddington 近似方法(Kiehl et al., 1996)。

### 3.2.2陆地表面模型(Land Surface Model)

（1）BATS

地表物理模块[默认]{.mark}使用Biosphere-Atmosphere Transfer Scheme
version 1e (BATS1e) Dickinson et al. (1993)。

Giorgi et al.,
2003a对BATS模型做了修正，使用Mosaic-type方法考虑了地形的压格子尺度变化。增加2种新的土地利用方式来表征城市和非城市环境，。。。

BATS模型的土地利用类型的参数化使用Kueppers et al. (2008)中的表1.

（2）CLM模型

[CLM]{.mark}陆地表面模型（可选）

CLM_v3.5的陆地表面参数化比较分析可参考Steiner et al., 2009

CLM更详细的物理参数化介绍见Oleson (2004)

### 3.2.3行星边界层格式(Planetary Boundary Layer Scheme)

Holtslag et al. (1990) **PBL**

UW紊流封闭模型

### 3.2.4对流降雨格式(Convective Precipitation Schemes)

对流降雨计算有3种方法可选：

\(1\) Modified-Kuo scheme Anthes (1977);

\(2\) Grell scheme Grell (1993);

\(3\) MIT-Emanuel scheme (Emanuel, 1991; Emanuel and Zivkovic-Rothman,
1999).

### 3.2.5大尺度降雨格式(Large-Scale Precipitation Scheme)

[Subgrid Explicit Moisture Scheme
(SUBEX)]{.mark}处理非对流云和模型求解的降雨。

SUBEX考虑云的subrid变化，by linking the average grid cell relative
humidity to the cloud fraction and cloud water following the work of
Sundqvist et al. (1989).

SUBEX还考虑雨滴的聚集和蒸发。

### 3.2.6新的云微物理格式

新格式基于ECMWF的IFS (Tiedtke \[1993\], Tompkins \[2007\]), Nogherotto
et al. \[2016\])。

新格式中添加了：

-   Liquid and ice water content are independent, allowing the existence
    of supercooled liquid water and mixedphase cloud.

-   Rain and snow now precipitate with a fixed, finite, terminal fall
    speed and can be then advected by the three dimensional wind.

-   The new scheme solves implicitly 5 prognostic equations for water
    vapor, cloud liquid water, rain, ice and snow. It is also easily
    suitable for a larger number of variables. Water vapor qv, cloud
    liquid water ql , rain qr, ice qi and snow qs are all expressed in
    terms of the grid-mean mixing ratio.

-   A check for the conservation of enthalpy and of total moisture is
    ensured at the end of each timestep.

各变量的控制方程是：

### 3.2.7海面通量参数化(Ocean flux Parameterization)

BATS，标准的Monin-Obukhov相似性关系，计算非常稳定条件下，没有特殊处理对流项的海气通量。另外，粗糙长度设为常数值，即不是风和稳定性的函数。

Zeng格式，描述所有稳定条件，包括考虑边界层尺度变化引起的额外通量的阵风风速。

### 3.2.8预测的海表温度格式

RegCM_v3中默认使用每6小时预设的海表温度(SST)，即使用来自周或月的SST再分析产品的时间插值。

RegCM_v4默认使用Zeng (2005)的海表温度模型。

### 3.2.9压力梯度格式(Pressure Gradient Scheme)

2种方法计算压力梯度力。

### 3.2.10湖泊模型 {#湖泊模型 .标题3}

Hostetler et al. (1993)

### 3.2.11Aerosols and Dust (Chemistry Model)
Marticorena and Bergametti (1995)

Alfaro and Gomes (2001)

# 参考文献

Grell, G. A., J. Dudhia, D. R. Stauffer, Description of the fifth
generation Penn State/NCAR Mesoscale Model (MM5), Tech. Rep. TN-398+STR,
NCAR, Boulder, Colorado, 1994.

Oleson K., et al.., Technical description of the Community Land Model
(CLM), Tech. Rep. Technical Note NCAR/TN-461+STR, NCAR, 2004.
[CLM_v3.5]{.mark}
