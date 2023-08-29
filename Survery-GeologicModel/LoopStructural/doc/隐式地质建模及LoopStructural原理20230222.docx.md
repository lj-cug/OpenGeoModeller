# LOOP3D GMD-special issue论文学习

## 简介

LOOP3D项目，是由西澳大利亚大学、Geoscience Australia,
加拿大地质调查局，德国亚琛工业大学等联合研发项目，旨在建立数字化地质建模的自动工作流程序。类似的项目有GemPy项目。

2021\~2022年，GMD期刊专辑介绍了LOOP3D项目系列论文和程序。包含：

[(1)]{.mark} LoopStructural 1.0: time-aware geological modelling

3D地质建模库，允许使用多种不同算法，比较相同构造地质建模。使用构造地质概念与技术建立地质模型，可以表征复杂构造，如褶皱和断层。

作者：Lachlan Grose

[(2)]{.mark} Modelling of faults in LoopStructural 1.0

岩石中的不连续断层表征移动了的两块岩石，考虑到地质模型中是有挑战性的，因为岩石单元断层的几何，不仅由不连续位置，还由断层的动力学特性，来定义。LoopStructural是一个构造地质学框架，通过在模型的数学框架内直接加入动力学模块，将断层考虑入地质模型。

作者：Lachlan Grose

[(3)]{.mark} Automated geological map deconstruction for 3D model
construction using map2loop 1.0 and map2model 1.0

可以从未修改的数字地图提取相关信息，可使用相关数据集自动化建立3D地质模型。通过自动化流程排除人为主观因素，使工作流可重复。

作者：Mark Jessel

\(4\) Spatial agents for geological surface modelling

稀疏数据建立复杂的区域性地质模型。基于代理的(agent-based)地质构造群集(swarming)行为是推动地质建模的关键方法。

作者：Eric de Kemp；加拿大地质调查局

\(5\) Structural, petrophysical, and geological constraints in potential
􀀁field inversion using the Tomofast-x v1.0 open-source code

使用地质模型和岩石特性的测量，从[地球物理数据]{.mark}（重力场异常,
磁场异常）模拟地球地下结构的不同方法。

作者：Jeremie Giraud

\(6\) dh2loop 1.0: an open-source Python library for automated
processing and classification of geological logs

从遗留钻孔数据集或不同钻探作业，提取和标准化钻孔信息。还可以升尺度(upscale)岩石信息，还可能使用这些功能开发储积物(thesauri)来识别和分组地质术语。

作者：Ranee Joshi

\(7\) Blockworlds 0.1.0: a demonstration of anti-aliased geophysics for
probabilistic inversions of implicit and kinematic geological models

通过传感器数据建立地质模型，特别是使用系列事件描述某区域的历史的模型。本方法保证模拟的地质特征的较小变化，诸如两个岩层之间的边界位置，不会导致传感器测量的不实际的大变化。

\(8\) loopUI-0.1: indicators to support needs and practices in 3D
geological modelling uncertainty quantification
（3D地质建模的不确定性定量评估）

采矿工程业主启动的调查结果表明：由于缺乏数据，很难实施不确定性的定量化、耗时、解译跟踪的低质量和不确定性定量性的相对复杂性。为缓解后者问题，loopUI提供当地和全局指标的开源集，测量不同地质模型之间的地质不确定性。

作者：Guillaume Pirot

# LoopStructural 1.0 （断层建模工具）

[本文侧重软件原理介绍。]{.mark}

没有合理考虑断层动力学或断层表面的观测，创建有断层地质单元的3D地质模拟是有挑战的。

多个断层相互作用的几何体，其中断层表面几何显著偏离平面，稀疏数据难以表征地质界面。

有2种方法在地质表面建模中考虑断层：

（1）在表面处考虑断层位移，但不考虑断层动力学，大多数情况是生成地质上不可预料的结果，诸如shrinking
intrusions, fold hinges without offset and layer thickness growth in
flat oblique faults；

（2）构建没有断层的连续表面，然手对连续表面施加动力学的断层计算，创建断层位移。

2种方法各有优点，但都不能捕捉到复杂断层网络情况下断层-断层的相互作用，诸如：fault
duplexes, flower structures (花状构造), listric faults
(铲形断层)，因为他们要么（1）施加的断层滑动方向是错误的（不是数据定义的），要么（2）需要大量采集数据来描述断层表面位置。

LoopStructural将断层动力学与隐式断面建模结合，通过使用断层动力学重构观测数据，在插值到断层面之前构建模型区域。这种新方法构建的模型与断层面观测数据和断层动力结果一致。直接整合断层动力学与断层面隐式建模，允许构建复杂的断层地层和模拟断层-断层相互作用。

LoopStructural在捕捉断层面几何方面有显著改进，特别是断层面之间相交角度和断面层变化的情况（如入侵，褶皱序列），以及模拟相互作用断层的情况（fault
duplexes）。

## 1 引言

在描述地质面时考虑断层，向隐式函数加入一个step function (de la Varga
andWellmann, 2016)，step
function在不连续的断层界面处，在标量场中施加一个跳跃，但限制是：断层面之间的角度不变，断层滑动方向不变，建模的断层不能真实代表断层移动，地质描述中没有使用step
function考虑断层动力学过程（移动方向和移动幅度）。

GemPy: Bayesian逆问题：断层位移的概率分布由断层位移观测得到(de la Varga
and Wellmann, 2016; Wellmann et al., 2017; Godefroy et al.,
2018)。但是没有在step function中考虑断层动力学过程。

LoopStructural考虑断层动力学过程（滑动方向和幅度）约束断层位移，对断层面实施断层计算(Godefroy
et al., 2018; Laurent et al., 2013; Georgsen et al., 2012)。

step function方法建立断层网的地质模型也是困难的，诸如splay faults,
cross-cutting faults, abutting
faults，需要合理地考虑断层间的相对动力学过程，特别是断层面观测很有限的情况。

LoopStructural采用Laurent et al. (2013) and Godefroy et al.
(2018)的方法，在隐式建模函数中直接考虑断层计算，使用断层动力学重构模型域和对pre-fault几何构建数据点。重构函数在[构造框架(Grose
et al.,
2021a)]{.mark}中定义，其中使用坐标定义断层面、断层滑动方向和断层的范围，这种time-aware地质建模是[承接]{.mark}褶皱建模(Grose
et al., 2017; Laurent et al., 2016)的工作流。

[示例]{.mark}：简单的断层入侵（展示step function的局限性）、a thrust
duplex system, a faulted fold series using the fold modelling framework
from Laurent et al. (2016) and Grose et al. (2017)。

## 2 三维建模方法介绍

### 2.1表面表征 {#表面表征 .标题3}

三维构造地质模型就是表征地下地质结构，其中地质单元或者使用边界面（上下接触面）(Wellmann
and Caumon, 2018)，或者是预设的支撑体(support)的离散。

地质建模的2种方法：

（1）显式建模(explicit modeling)：the geometry of surfaces are contained
using a [support]{.mark} that is collocated with the surface geometry.
使用离散对象，如三角化表面、2D网格或参数化表面表征这些界面。通常，或者使用三角化数据点，或者使用插值算法创建光滑面拟合数据，来构建几何界面。显式界面表征意味着界面几何仅在界面位置处描述。

（2）隐式建模(implicit
modeling)：使用3D空间中的一个或几个标量场的等值面来表征地质界面的几何形体（如stratigraphic
horizons and fault surfaces）(Lajaunie et al.,
1997)。标量场的值表征距离一个参考平面的距离。也就是说，如果地质界面是整合接触的(conformable)，由表征界面间相对厚度的标量场的等值面来描述。标量场的梯度是待建模表面的方位(orientation)。使用不同的插值方法构建这些标量场，如co-Kriging,
径向基函数或在预定义的support上使用离散插值。

隐式建模偏差更小，可重复性更高，还可能生成反映地质不确定性的一套地质模型。

### 2.2隐式建模中的断层 {#隐式建模中的断层 .标题3}

错位的地质特征（stratigraphic interfaces, faults or
foliations）几何表征，应该与断层的动力学过程一致。

有3种方法（见图1），用来在隐式界面描述中考虑断层：

\(1\) interpolate fault domains using independent implicit functions

\(2\) incorporate the fault into the domain discretisation

GemPy: Baysian推理，优化模型参数，获取符合地质知识和地质观测的拟合模型。

\(3\) apply a fault operator to a surface already interpolated

通过使用数值算子，定义在断层面或附近的分布，在隐式三维建模中考虑断层动力学机制。

还有一个trishear模型常与地震解释结合使用，是一次性构建所有断层，而顺序地重构断层允许考虑断层拓扑关系。

### 2.3已有方法的局限性 {#已有方法的局限性 .标题3}

上述的已有地质建模方法都不能考虑断层动力机制，也不能将地质观测考虑到地质界面描述。

Step
function方法是有吸引力的，因为模型描述中考虑了断层，但不能捕捉某些断层的动力学过程，如：the
angle of intersection between the fault and faulted surface is variable
(e.g. fold series, or intrusions)。

如图2a和图2c，褶皱序列和侵入体被逆断层错位。图2b和图2d显示了使用step
function考虑断层。

## 3 断层建模的动力学框架

修改Laurent et al. (2013)和Godefroy et al.
(2018)的工作流，在隐式表征地质特征中考虑断层算子，如图3。算法流程：

1\. building the fault frame, a curvilinear coordinate system
representing the fault geometry

2\. defining the fault displacement within the model domain

3\. adding the fault kinematics to the implicit surface description

### 3.1断层框架 {#断层框架 .标题3}

曲线坐标系统（断层框架）

第一个坐标(*f*~0~)表征距断层面的距离，0等值线表示断层面，可从断层面的观测数据插值得到，如strike和dip控制和断层面位置的观测（图4a）。

第二个坐标(*f*~1~)度量断层的位置方向，该场的梯度法向将与断层的任意动力学指标并行（即slickensides,
stretching lineations）且与断层面也平行。

第二个坐标(*f*~2~)度量断层范围方向上的距离（图4b中的绿线），该场的梯度方向与断层面(*f*~0~)的梯度法向正交，也与断层位移场(*f*~1~)正交。

3个局部方向向量由任意位置的断层框架的归一化梯度隐式定义：

![](./media/image1.emf){width="1.1066819772528433in"
height="0.6961373578302712in"}

然后，可在任意位置查询模型内的断层框架，返回距断层中心的距离以及断层框架向量。

### 3.2断层位移 {#断层位移 .标题3}

由*f*~1~表征的断层位移方向。断层位移幅度，或者由hanging
wall的常数位移定义，或者通过断层框架坐标定义。

我们对任意位置处的局部断层位移定义为：

![](./media/image2.emf){width="0.9837981189851268in"
height="0.24079286964129484in"}

式中，*d*为某位置上的断层位移幅度的标量值。

### 3.3 时间感知的地质建模 {#时间感知的地质建模 .标题3}

在计算标量场数值之前，对数据点实施断层重构动力学算法：

![](./media/image3.emf){width="2.53582239720035in"
height="0.22504702537182852in"}

使用这个工作流，可以堆叠实施多个断层算子，并时间反向地实施适当的位移，在错位之前重构其位置。

这种方法可以应用于任意构造单元，包括：lineations, fold vergence,
tectonic foliations, fold frames (Grose et al., 2017)。

## 4实施

一般的工作流与插值格式无关。使用LoopStructual的离散插值算法构建断层框架。使用离散插值算法，通过求解support定义插值问题的复杂度，而不是通过约束的数目。

### 4.1构建断层框架 {#构建断层框架 .标题3}

使用下列流程，通过插值标量场，构建断层框架：

1\. Interpolate coordinate 0 to represent the geometry of the fault
surface so that the isosurface of 0 contains the fault trace and the
field is parallel to the orientation of the fault surface.

2\. Interpolate coordinate 1 so that the direction of its gradient norm
is orthogonal to the direction of the gradient norm of the fault surface
and parallel to any kinematic indicators for the fault.

3\. Interpolate coordinate 2 so that the direction of its gradient norm
is orthogonal to the direction of the gradient norm of the fault surface
field and to the fault slip direction field.

### 4.2三维断层位移 {#三维断层位移 .标题3}

对归一化的断层框架坐标实施3个函数来定义断层滑动Godefroy et al. (2018)。

![](./media/image4.emf){width="2.683266622922135in"
height="0.2292082239720035in"}

### 4.3 Splay faults {#splay-faults .标题3}

## 5案例研究

基于LoopStructural 1.2.0实施。

### 5.1断层入侵 {#断层入侵 .标题3}

一个合成的侵入体被一个平面的断层给错位。

### 5.2 Finite fault {#finite-fault .标题3}

### 5.3 Thrust duplex {#thrust-duplex .标题3}

### 5.4 Faulted fold series {#faulted-fold-series .标题3}

## 6讨论

## 7结论

代码：本文使用的是1.2.0版本。最新的LoopStructural 1.5.4

安装：pip install LoopStructual

示例（包括数据集）使用Jupyter notebook, Zenodo

[lachlangrose/grose_et_al_2021_gmd_faults-1.1.zip](https://zenodo.org/record/5234634/files/lachlangrose/grose_et_al_2021_gmd_faults-1.1.zip?download=1)

## 参考文献

Lachlan Grose, et al. Modelling of faults in LoopStructural 1.0. Geosci.
Model Dev., 14, 6197-6213, 2021.

Grose, L., Ailleres, L., Laurent, G., and Jessell, M.: LoopStructural
1.0: time-aware geological modelling, Geosci. Model Dev., 14,
3915--3937, 2021a.

# LoopStructural 1.0: 时间感知的地质建模

[本文侧重软件架构和实战的介绍。]{.mark} [time-aware: 时间感知的]{.mark}

本文介绍6种插值算法，包括：3个离散插值算法和3个多项式趋势插值算法，用于相同的建模设计。表明：可对不同的地质对象，如conformable
foliations,断层面和不整合，混合使用不同的算法。使用时间感知方法在建模中考[虑地质特征，其中最现代的特征最先建模，用来约束更古代的地质特整体的几何建模。]{.mark}

## 1 引言

不同地质特征之间的拓扑关系，诸如：horizons, faults interactions,
intrusions and unconformities，对不同的模型组件使用多种隐式函数。

隐式建模有2种方法：

（1）数据支持的方法：在数据点处，计算基函数(Lajaunie et al., 1997)

（2）离散插值，基函数在预定义的support上。

商业软件（如Petrel,
Leapfrog等）使用的建模算法，一般仅提供一种插值算法，难以比较不同插值格式的效果；并且是"黑盒子"算法，修改算法参数的功能有限，无法了解算法是怎么实施的。最近，开源的GemPy库
(de la Varga et al.,
2019)使用高性能计算库，实施双co-Kriging隐式插值算法。

LoopStructural，基于Laurent et al. (2016) and Grose et al. (2017, 2018,
2019)的不断贡献，实施3D地质建模。核心库依赖于SciPy, numpy,
pandas和一些科学计算的Python库；可视化模块使用LavaVu (Kaluza et al.,
2020)，一个小型的OpenGL可视化软件包，允许在Jupyter
Notebook环境中可视化模型。

算例：

第3个算例：使用前处理模块map2loop，准备了Flinders
Ranges地区的实际数据集。

第4个算例：使用map2loop准备的Hamersley地区的输入数据集。

## 2材料与方法

3D地质建模软件的2个主要功能：

（1）从地质观测和地质知识创建表面，称为插值；

（2）对表面描述引入地质概念，例如：断层面应显示位移，不整合应显示为单元之间的边界。

LoopStructural的面，使用一个或多个volumetric scalar
fields的等值面隐式表征。

以时间感知的方法，通过增加地质事件（folding event, one fault, another
fault, an unconformity）的构造参数，管理地质准则。

### 2.1隐式界面建模 {#隐式界面建模 .标题3}

隐式表面建模使用函数f(xyz)表征地质特征的几何。使用2种方法：（1）使用局参考水平面的距离作为标量场；（2）势场方法。

隐函数使用基函数的加权形式表征：

![](./media/image5.emf){width="1.7778893263342082in"
height="0.48031496062992124in"}

有2种方法近似隐函数：（1）使用离散公式的插值，其中*N*定义为某种网格；（2）数据支持的基函数方法，其中*N*为数据点的数目。

2.1.1输入数据

在3D建模中考虑地质观测可分为2种：

（1）描述地质特征方位(Orientation)的观测，如：on contact and off contact

（2）描述地质特征位置的观测（cumulative thickness for conformable
stratigraphic horizons, or location of fault surface）

在地质图中，location observations may be the trace of a geological
surface on the geological map, or a single point observation at an
outcrop or from a borehole.

Orientation observations generally record a geometrical property of the
surface -- e.g. a vector that is tangential to the plane or the vector
that is normal to the plane (black and dashed arrows in Fig. 1).

图1
显示不同类型插值约束的示意图，在2D上实施隐式插值格式。有2个界面：值为0的参考面，值为1的临近界面。这里展示了3类约束：(1)
标量场法向约束标量场的方位以及该位置处的隐函数的范数；(2)标量场值约束控制标量场的值；(3)切向约束仅约束隐函数的方位，不约束范数。修改自Hillier
et al. (2014)

当使用势场方法建模时，通过法向控制点的大小推测标量场。

使用符号距离方法时，观测值定义标量场，有效控制层厚度。

方位约束，要么控制方位的分量，即：指定函数的梯度应该与观测点以及隐函数的梯度法向正交。

在模型中的一个位置处，所有地质观测约束隐函数的一个分量：

（1）地质特征位置的观测将约束标量场*f*(*x,y,z*)=*v*的值。

（2）接触方位观测可以：

1\) 约束函数的偏微分![](./media/image6.emf){width="1.0984120734908136in"
height="0.1915518372703412in"}

2\)
约束与接触平行的一个向量![](./media/image7.emf){width="1.2276257655293088in"
height="0.20026793525809275in"}

2.1.2 分段线性插值

由在3D四面体网格上分段线性函数定义体标量场。LoopStructural创建3D四面体网格是通过分割一个规则笛卡尔网格为四面体网格，其中一个立方体分为5个四面体（见[附录A]{.mark}）。

2.1.3有限差分插值

使用笛卡尔网格上的tri-linear基函数，近似插值函数。基函数将插值函数描述为单元角的函数，在角点上计算函数，见附录B的tri-linear基函数。

2.1.4求解离散插值

使用分段线性插值算子或有限差分插值算子，由support的节点值定义标量场。通过求解有***M***个未知量的方程组得到标量场值。

**A**x=**b**

LoopStructural使用SciPy的共轭梯度法求解线性方程组。

2.1.5数据支持的插值

使用相同位置处的基函数作为数据点做隐式面建模。

LoopStructural使用SurfE,
C++程序，实施通用型径向基函数插值。SurfE有3种隐式表面重构：（1）使用径向基函数的符号距离插值；（2）使用双Co-Kriging的势场插值；（3）对各个面使用独立的标量场的符号距离插值。

LoopStructural与SurfE的接口允许用户使用所有的插值参数，这也包括了可使用更多复杂的求解器以及插值中添加光滑参数。

### 2.2模拟地质特征 {#模拟地质特征 .标题3}

在地质模型中，岩石几何的软件相互作用，有3种方法：

（1）分层接触\--沉积层之间的接触

（2）断层接触

（3）侵入接触

这些地质界面可以被变形构造影响，诸如folds, faults and shear zones

2.2.1沉积层接触(Stratigraphic contacts)

图2 不整合接触（红线）和地质界面（黑线）表征沉积历史的间断

一个不整合接触有不同可能的几何：(a) disconformaty; (b) angular
unconformity; (c) nonconformity

2.2.2构造框架

图3 (a) 构造框架显示3个坐标上的等值面；(b) 表征一个褶皱的构造框架; (c)
表征断层几何体的构造框架

2.2.3断层

图4 断层位移剖面: (a) 恒定位移剖面; (b)
无穷范围的断层位移，显示沿断层范围或在滑动方向上断层位移没有变化; (c)
固定范围的断层位移，显示沿断层滑动方向的位移减小；(d)
固定范围钟型断层位移剖面，保证沿断层范围方向或滑动方向上的断层未有

2.2.4褶皱

图5 褶皱: (a) 褶皱框架; (b1) 褶皱框架方向矢量; (b2)
褶皱轴向旋转角度定义的轴; (b3) 绕褶皱轴旋转的褶皱臂定义的褶皱foliation

## 3 LoopStructural的实施

### 3.1 Loop结构设计 {#loop结构设计 .标题3}

Python3.6+，使用Numpy数据结构和操作。

[5个]{.mark}子模块：

-   core

-   interpolation

-   datasets

-   utils

-   visualization

初始化实例，需要的参数：建模盒子边界的最大和最小范围，定义为2个独立向量，rescale系数：

![](./media/image8.emf){width="3.23667104111986in"
height="0.2311920384951881in"}

通过使用GeologicalModel的实例，增加不同的地质对象。插值算法中可考虑4种不同类型的观测：

（1）value

（2）gradient

（3）tangent

（4）norm

与GeologicalModel相关的数据使用set_data
(data)方法，其中data是pandas数据框架。向模型添加数据点时，将转换为模型的坐标系统。

### 3.2 添加地质对象 {#添加地质对象 .标题3}

GeologicalFeatures表征所有的地质对象，包括：stratigraphy, faults,
folding event,
unconformities。一个GeologicalFeatures可以根据标量场的值and/or在某个位置上的标量场的梯度计算得到。

GeologicalFeatures包含顺序的地质特征的集合，确定这些地质特征如何相互作用。

GeologicalFeatures添加入GeologicalModel采用不同的方式，这要根据要模拟的地质特征类型。

LoopStructural可以在相同模型中为不同的GeologicalFeatures指定不同的插值算法。通过为函数增加额外的关键词参数，指定插值算法和任何参数定义。[表1]{.mark}罗列了定义插值算法的可能参数。

图6 展示2个等值面使用不同插值算法生成的合成表面：(a)输入数据; (b)
使用PLI插值的表面; (c) 使用FDI插值生成的表面; (d) 使用SurfE插值的表面

### 3.3模型输出 {#模型输出 .标题3}

在地质模型的坐标矩阵上计算GeologicalModel。可以从一个GeologicalModel调用如下函数：

（1）为计算某位置处的岩性值(lithology
value)，函数evaluate_model(xyz)返回包含层序编号的整型ID的numpy数组，定义为层序列。

（2）为计算某位置处的GeologicalFeature的值，函数evalute_feature_value(feature_name,
xyz)返回表征地质特征的标量场的值。

（3）为计算GeologicalFeature的梯度，可以调用evaluate_feature_gradient(feature_name,
xyz)。

[三角化表面]{.mark}可以从GeologicalFeature提取，然后输出到常用的网格格式，如：VTK
(.vtk)或Wavefront (.obj)。这些表面可以输入到外部软件，如ParaView。

### 3.4模型可视化 {#模型可视化 .标题3}

LoopStructural有3种可视化工具（LoopStructural.visualization模块）：

（1）LavaVuModelViewer：LavaVu (Kaluza et al.,
2020)交互式可视化，使用LavaVu显示三角化表面，表征地质界面及描述隐函数的标量场。

（2）MapView：使用matplotlib做2D可视化（断面，地图），从得到的地质模型创建一个地质图。使用strike和dip符号绘制接触的位置和方位。在地图面上计算标量场，绘制等值线或将地质模型绘制到地图上。

（3）FoldRotationAnglePlotter：生成S制图和S-variogram制图的可视化模块，展示褶皱地质特征。使用matplotlib绘制。

## 4示例

### 4.1隐式表面建模 {#隐式表面建模 .标题3}

模拟2个合成表面，使用相同标量场，模型体积(-0.1, -0.1, -0.1)和(5.1, 5.1,
5.1)。

2套观测数据点。第1套在*z*=4处的规则网格上形成一个表面，第2套在![](./media/image9.emf){width="3.047399387576553in"
height="0.14865266841644795in"}处形成一个表面。

图7a显示数据点，图7b、图7c和图7d显示使用3种默认的插值算法生成的表面(PLI,
FDI, SurfE)。

当使用离散插值时，调整约束的权重通常对建模结果影响最大。如图8，......

### 4.2模拟褶皱：类型3推理 {#模拟褶皱类型3推理 .标题3}

图8a显示相对F2, S2轴向foliation

### 4.3与map2loop的整合 {#与map2loop的整合 .标题3}

使用map2loop (Jessell et al.,
2021)作为前处理程序，从区域性地质调查图、国家地层数据和全球DEM，生成输入数据集。

map2loop创建一套augmented数据文件，用于在LoopStructural中构建地质模型。

类方法GeologicalModel.from_map2loop_directory(m2l_directory,
\*\*kwargs))，从一个map2loop输出根目录，创建一个GeologicalModel实例。

下面通过2个实际案例展示map2loop与LoopStructural间的接口：

（1）Flinders Ranges in South Australia

（2）Hamersley region in Western Australia

第1个案例展示对大的区域性模型map2loop与LoopStructural间的接口。

第2个案例展示如何修改生成输入数据集的概念模型。

第1个案例：模型面积85 km x 53
km。图12a展示层序单元；图12b展示地质图中的露头地质，没有任何地质单元的地图表示浅层Tertiary与Quaternary覆盖。map2loop从露头地质单元提取基部接触，计算分层厚度。

建模工作流都封装在GeologicalModel.from_map2loop_directory(m2l_directory,
\*\*kwargs))类方法，意味着无需任何用户输入就能生成地质模型。

生成的地质模型如图13，表面表征一个层序组的基底，使用分层列着色（图12a）。模型中的断层使用50000个单元的笛卡尔网格插值，使用FDI插值算法，插值矩阵使用pyamg多重网格求解器。层序使用更细的网格（50万个单元），使用FDI插值及pyamg。i7处理器的笔记本（32Gb内存），使用map2loop花费近1min处理数据，渲染笛卡尔网格(200x2000x100)的表面花费3min。

第2个案例：使用map2loop处理小范围的Turner Syncline in the Hamersley
region inWestern Australia，数据由Geological Survey of Western
Australia提供。模型范围12km x
13km，包含3个断层。map2loop默认假设断层是垂直的、纯dip滑动、通过分析断层单元的地图模式计算hanging
wall与位移（参考Jessell et al., 2021）。

图14显示生成的地质模型，包括断层滑动向量和断层平面。

## 5讨论

（1）时间感知的地质建模，采用与地质事件发生的相反顺序添加地质特征，这样能捕捉复杂构造几何，这种方法用于模拟refolded褶皱（图11）。

（2）LoopStructural提供了灵活的开源实施隐式地质建模算法工作流。这意味着可以开发新的隐式建模算法与工具，比如从外部网格生成代码导入LoopStructural，使用一个二次开发的类重写四面体网格类。

（3）反问题，模拟地质建模中的不确定性。

（4）map2loop从开放的地质调查数据（层序数据集、DTM、地质shapefile、构造线和构造观测）生成增强数据集。从数据处理到模型渲染需要约10min。

（5）断层位移剖面的优化。断层滑移向量和断层倾角由概念模型或观测数据定义，但是指定约束断层滑移的具体的概率函数是有挑战的，与具体的地质问题有关，即当观测数据充足时，可使用Godefroy
et al.
(2018b)???的方法并从插值中分离部分数据；但是当处理典型的区域性比例尺地图层时，大部分观测仅在地表附近，约束3D地质几何是有限的。

## 6结论

## 附录A 四面体网格

一个立方体定义为8个节点，使用编号ijk定义在笛卡尔网格。

单元内的线性插值属性为：

四面体内部属性可使用4个如下的形函数做插值：

四面体内的函数梯度计算如下：

## 附录B 立方体单元内的trilinear插值

隐式函数，可使用形函数(N~0,...7~)相对单元的8个节点描述：

## 参考文献

Kaluza, O., Moresi, L., Mansour, J., Barnes, D. G., and Quenette, S.:
lavavu/LavaVu: v1.6.1, Zenodo, https://doi.org/10.5281/zenodo.3955381,
2020.

Godefroy, G., Caumon, G., Ford, M., Laurent, G., and Jackson, C. A.-L.:
A parametric fault displacement model to introduce kinematic control
into modeling faults from sparse data, Interpretation, 6, B1-B13,
ttps://doi.org/10.1190/INT-2017-0059.1, 2018b.
