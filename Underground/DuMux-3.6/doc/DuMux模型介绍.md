# DuMux模拟框架

[德国Stuttgart大学，DFG资助]

摘要：DuMux ver. 3，基于模块化C++框架DUNE (Distributed and Unified
Numerics
Environment)，改善透明性和开发效率及社区建设，着眼于模块化和可重复使用。DuMux
ver.
3引入更一致有限体积格式抽象。最后，介绍了multi-domain模拟的新框架。通过3个数值例子展示其灵活性。

## 1引言

DuMu^x^, Dune for multi-{phase, component, scale, physics, domain, . . .
} flow and transport in porous
media，模拟多孔介质章的流动与传输的开源模拟器(dumux.org)。

已成功应用于CO~2~封存(2-5)，放射性废物处理(6)，天然气迁移问题(7)，环境修复问题(8)，裂隙空隙介质中的流动(9-14)，多孔介质中的反应输移(15)，生物薄膜和矿物沉积模拟(16,
17)，土壤-植物根系相互作用模拟(18,
19)，燃料单元模拟(20)，多孔网络模拟(21)，地下-大气耦合模拟(25, 26)。

DuMu^x^基于DUNE框架，DUNE核心模块提供：多种网格管理，提供灵活的网格接口；线性代数抽象，迭代求解器后端(31)，实施并行计算的抽象。

DuMu^x^的特点是：多相和多组分流体和输移模型的综合库，流体与材料的本构关系框架的模块化，有限体积离散格式的抽象化（第3节），着眼于模型耦合（第4节）。

已有很多开源项目模拟Darcy尺度的多孔介质流动与输移过程，列表见(L. Bilke,
2019)。

最常用的有：MODFLOW, MRST, OpenGeoSys, OPM, ParFlow, PFoTran, PorePy

还有开源的数值软件框架，如deal.II, DUNE, Feel++, FEniCS, MOSE, OpenCMISS

本文是B. Flemisch et al., 2011的后续，介绍了2.x版本的DuMux (B. Flemisch
et al., 2011)。

### 1.1发展历史 {#发展历史 .标题3}

DuMux项目起始于2007年1月，德国Stuttgart大学的Department of
Hydromechanics and Modelling of Hydrosystems，最后在2009年发布DuMux
1.0,许可证是GNU GPL2

在2011年2月发布DuMux 2.0，发表论文B. Flemisch et al., 2011

2013年10月，发布DuMux 2.X

2017年12月，发布DuMux 2.12

从2.7版本，使用zendo指定发布版本的DOI及tarball

由于兼容问题，代码开发从2016年11月，由一个小组开始重新组织，以独立开发线研发代码。改变的细节信息见第2节。由于大量代码修改，维护向后兼容变得不可行。因此，启动了新的major发布循环。

2018年12月，发布了DuMux 3.0

2019年10月，发布了DuMux 3.1

### 1.2作为一个框架的DuMux {#作为一个框架的dumux .标题3}

DuMux的git代码仓库：git.iws.uni-stuttgart.de

代码质量控制与重复性，引入2个模块：dumux-course和dumux-lecture

DuMux与OPM项目相关，着眼于开发模拟工业和科学相关的多孔介质流动与输移过程。黑油模拟器Flow是OPM的主要产品(A.
Fl Rasmussen,
2020)，构建于OPM模块opm-material与opm-models，这些模块基于DuMux
2.2的fork。DuMux与OPM模块opm-grid合用，使得DuMux可使用角点网格，是石油工业的标准。

A. Fl Rasmussen, T. Harald Sandve, K. Bao, A. Lauser, J. Hove, B.
Skaflestad, R. Klkorn, M. Blatt, A. Birger Rustad, O. Sareid, K.-A. Lie,
A. Thune, The open porous media flow reservoir simulator, Comput. Math.
Appl. 81 (2020) 159-185.

## 2结构与设计原则

DuMux设计为研究型代码框架，强调模块化(modularity)。以一个独立的DUNE模块开发DuMux，这意味着对于使用者，DuMux的所有模拟组件可以很容易用新的实施模块来取代，无需修改DuMux模块本身的代码。例如，修改数值通量计算、将单元中心转为节点中心的有限体积离散。DuMux得益于DUNE的设计。例如，仅改变用户代码的一行，就可以实现将非结构网格实施变为结构网格实施。

通过C++模板和范式编程

面向对象编程设计准则

算法与数据结构分离，例如DuMux的材料框架设计。

DuMux 3需要C++-14兼容的编译器，使用C++的现代特征，如lamda, smart
pointer，增加DuMux的usability, modularity, and efficiency

DuMux环境下，PDE耦合系统称为model，包括需要封闭的本构方程。很多模型描述多孔介质中的non-isothermal,
multi-component multi-phase flow processes

DuMux的主要组件采用C++ class

Problem class定义边界条件、初始条件、体积源汇项，通过定义的类接口

SpatialParams类，定义材料参数的空间分布，如孔隙度和渗透系数，或例如van
Genuchten水滞留模型的参数。

Code Example 1: DuMux 3的属性设置示例

Code Example 2:

## 3抽象与通用有限体积格式概念

DuMux的最重要抽象是[网格几何]，这个概念在DuMux 2已有，在DuMux
3的面向对象表征得到重新设计。一个网格几何是以grid view对DUNE
grid实例的封装。DUNE网格是分级网格，网格视角提供read-only访问网格的某些部分。

下面介绍在网格几何概念背后实施数学抽象，以C++类的形式介绍数学方法的实现。

### 3.1有限体积离散

![](./media/image1..png)

### 3.2主单元整合

### 3.3软件实施中的表征

Code Example 3:

## 4 Multi-domain模拟（模型耦合）

多区域模拟已成功用于实施与流体和输移过程的耦合模拟，如vascularized brain
tissue \[23, 63\], root-soil interaction models in the vadose zone
\[18\], flow and transport models for fractured rock systems \[12,64\],
coupled porous medium flow and atmosphere flow (Darcy-Navier-Stokes) at
the soil surface \[55\], and a model that couples a pore-network model
with a Navier-Stokes model \[21\].

Code Example 4: 模型耦合couplingStencil

## 5 DuMux 3的新特性

DuMux ver.3相比ver.2增加了新特性。包括：

-   高级类抽象可读性提高，如assembler, linear and non-linear solvers,
    > grid readers and grid geometry, and file
    > I/O，更灵活的main函数（见第2节）。

-   很多模型在代码重利用和模块化方面都改善了，因此最小化代码重开发，改进可读性。

-   除了流体系统概念，引入了固体系统概念，便于增加新模型，如物质的矿化或沉降修改孔隙矩阵结构(17,
    > 65)。

-   可改变孔隙材料属性，如渗透系数和孔隙度，线性或非线性与主变量有关。

-   通用的热和化学非平衡模型，与任何多孔介质模型联合使用。

-   使用Maxwell-Stefan扩散模拟多组分扩散。

-   所有模型都可使用灵活实施的单元中心的MPFA-O格式。

-   重新设计Navier-Stokes模型，在交错网格上使用MAC格式(21, 55,
    > 66)，包括RANS模型(67, 68)及紊流模型(k-epsilon,
    > k-omega)及二阶迎风格式。

-   [现在可以求解浅水方程的问题。]

-   多区域模拟实现灵活的模型耦合（见第4节）。

## 6数值算例（DuMux 3）

挪威Norne油藏模拟：基于opm-data的两相流模拟。孔隙度场、各向异性的渗透率场、计算网格*M*的单元*E*、注水和产油井，如图4。计算域用角点网格表征，使用omp-grid的DUNE网格接口。

不可压缩、不融合两相流模型方程与参数参考文献(M. Schneider,
2018)的式56-60及表3.

M. Schneider, B. Flemisch, R. Helmig, K. Terekhov, H. Tchelepi, 2018.
Monotone nonlinear finite-volume method for challenging grids, Comput.
Geosci. 22 (2): 565-586.

非线性耦合PDE使用TPFA单元中心有限体积格式空间离散，向后欧拉格式做时间离散。初始条件是计算域完全是饱和石油。从2个井注水。2个抽水井开始产油，之后是水油混合物。

井使用固定的Bore-hole压力的Peaceman井模型来模拟。

湿相压力与饱和度的时间演变见图5。

DuMux
3的开发主要是模型耦合。下面给出3个数值算例展示新的multi-domain框架的灵活性（局限：不能并行计算）。本文3个数值算例的代码及数据见：git.iws.uni-stuttgart.de/dumux-pub/dumux2019

[算例1]是模拟多孔介质上的自由流动，通过耦合Navier-Stokes模型与Pore-network模型。计算域在公共界面上耦合。

[算例2]展示孔隙岩石矩阵中的两相流。裂隙流动在低维（剖面2D）计算域上模拟，与3D岩石矩阵计算域离散的控制体的面协调，可实现模拟高传导性的裂隙及不可渗透的裂隙。这个算例展示TPFA与MPFA-O有限体积格式模拟具有各向异性渗透性的岩石矩阵的不同。

[算例3]展示植物根系吸收(uptake)和跟踪物质输移的模拟。根系由tube网络嵌入土壤矩阵来表征。使用合适的源项模拟2个非协调(non-conforming)域的物质交换。

DuMu^x^的前后处理工具，建议使用外部软件建立workflow，如下：

-   网格划分：Gmsh

-   作图：Gnuplot, Matplotlib

-   可视化：ParaView

### 6.1算例1：自由流动与多孔网络模型的耦合 {#算例1自由流动与多孔网络模型的耦合 .标题3}

修改至：

K. Weishaupt, V. Joekar-Niasar, R. Helmig, An efficient coupling of free
flow and porous media flow using the pore-network modeling approach, J.
Comput. Phys. X 1 (2019) 100011,
<http://dx.doi.org/10.1016/j.jcpx.2019.100011>.

模拟通过2D计算域的随机多孔结构的非恒定过渡多组分流。明渠流控制方程为NS方程，忽略重力与稀释过程。

![](./media/image2..png)

各组份*k*的molar平衡方程：

NS方程使用交错网格上的MAC格式离散。Pore-network模型也在DuMux中实施，作为稳定的代码基座的一部分发布。空隙网络使用内嵌在2D计算域内的1D网络表述，使用DUNE网格的dune-foamgrid实施(80)。

O. Sander, T. Koch, N. Schrer, B. Flemisch, 2017. The Dune FoamGrid
implementation for surface and network grids, Arch. Numer. Softw. 5 (1):
217-244, <http://dx.doi.org/10.11588/ans.2017.1.28490>.

图6

图7

### 6.2算例2：裂隙多孔介质的两相流 {#算例2裂隙多孔介质的两相流 .标题3}

https://git.iws.uni-stuttgart.de/dumux-repositories/dumux-course/tree/master/exercises/exercise-fractures

模拟考虑浮力驱动的在初始完全水饱和的裂隙多孔介质中气态氮的向上迁移过程。

裂隙网络几何使用文献(84)，图8显示了边界条件。计算域为2D

图8

图9

### 6.3算例3：植物根系-土壤相互作用

模拟包括跟踪物输移的根系吸水过程。模型概念与数值方法介绍见文献(18)。DuMux的测试相对简单些。

## 目前的局限与前景

（1）设计[多种网格的multi-domain模拟目前不能并行计算]{.underline}。DUNE网格实施负责管理基于MPI的分布内存并行。使用2个或更多网格实例时，数据必须在网格间通信。

（2）DuMu^x^目前[仅支持向前和向后Euler时间离散]{.underline}。实施其他时间离散格式，需要refactor
assembly过程。

（3）基于Tag的C++编程访问模型的[property]{.underline}，tag作为模板形参（见FVAssembler,
Code Example
2），这明显[减少了一个类的模板形参的数目，妨碍了类的模块化与重利用性]{.underline}。解决方法：用显式依赖取代所有的这些类，[property技术]{.underline}也是[妨碍增加Python
wrapper的困难]{.underline}。

（4）DuMu^x^的重点是提供很多可用的和广泛的模型，[灵活的线性代数算法很少]{.underline}。[通过dune-istl提供]{.underline}有效的和灵活的数据结构，以及易用的和广泛的[预处理线性求解器]{.underline}。[目前DuMu^x^不便于线程方程组的代数操作]{.underline}，诸如[Jacobian矩阵的不同排序]{.underline}，[quasi-Newton格式]{.underline}或[复杂的基于矩阵分解的线性求解器]{.underline}策略。

（5）DuMu^x^的开发[继续反映多孔介质研究的最新进展]{.underline}。例如，改进free-flow和浅水模型与多孔介质流动模型的耦合求解；引入单相和多相流在pore-network模型，包括静态和动态方法；preCICE,
SusI

## 参考文献

Timo Koch, et al. DuMux 3--an open-source simulator for solving flow and
transport problems in porous media with a focus on model coupling.
Computers and Mathematics with Applications 81 (2021) 423-443.

B. Flemisch, M. Darcis, K. Erbertseder, B. Faigle, A. Lauser, K.
Mosthaf, S. Müthing, P. Nuske, A. Tatomir, M. Wolff, R. Helmig, DuMux:
DUNE for multi-{phase, component, scale, physics, . . . } flow and
transport in porous media, Adv. Water Resour. 34 (9) (2011) 1102--1112

L. Bilke, B. Flemisch, T. Kalbacher, O. Kolditz, R. Helmig, T. Nagel,
Development of open-source porous media simulators: Principles and
experiences, Transp. Porous Media (2019)
http://dx.doi.org/10.1007/s11242-019-01310-1.
