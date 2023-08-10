# Considerations on a generic water-bed exchange Module

## 1 Introduction

At present several packages already exist that can cope with multiple
sediment fractions and different bed layers: ROMS (Warner et al.
\[2008\]) Delft3D (Deltares), or MIKE (DHI). Furthermore, stand-alone
bed-modules are developed, e.g. Harris and Wiberg \[2001\], SEDTRANS (Li
and Amos \[2001\], and Neumeier et al. \[2008\]), Sanford \[2008\].
Although these modules have many things in common, they are all
different: some modules have a Lagrangian framework, others an Eulerian
one; some are developed for mixtures of sand/mud, others for
gravel/sand; some account for only two fractions (resulting with
continuity in only one fraction to be resolved), others for multiple
fractions; some consider biological mixing, others neglect this; some
allow consolidation, others not.

## 2 Framework

Aggradation and degradation can be treated in a Lagrangian framework or
in an Eulerian framework, see Figure 1.1. In the Lagrangian framework,
the thickness of the layers is constant and the set of layers moves with
the aggradation/degradation by means of an artificial advection
velocity, see Van Ledden et al. \[2004\].

![](./media/image1.emf){width="4.20842738407699in"
height="2.884313210848644in"}

Figure 1.1: Sketch of the change of grid for aggradation and degradation
for the Eulerian and Lagrangian framework.

Lagrangian模式下，床沙分层厚度不变，河床冲淤过程中仅通过人为设置对流速度，移动分层位置来计算冲淤厚度。Lagrangian模式的优缺点：

优点：网格本身不变，分层厚度是变化的。通常情况下，要求河床表层附近需要高分辨率网格，河床底部分辨率可以较粗一些。

缺点：计算精度与数值格式密切相关，分层之间的网格移动依赖于数值扩散系数，床沙分层可能被光滑，需要使用复杂的空间离散数值格式克服该缺陷。

Eulerian模式下，床沙分层位置不变，冲淤计算仅考虑表层厚度变化。发生淤积时，当表层（活动层）厚度太厚，将被分裂；发生冲刷时，表层厚度接近0，表层将与第2层合并。

缺点：河床表层没有固定的厚度，该层厚度会显著影响河床演变的时间尺度，因此期望表层厚度不变，或接近表层的数层厚度不变。

因此，需要融合Lagrangian模式和Eulerian模式。

## 3 Transport Formulation

将泥沙分为悬移质和推移质在粘性沙情况下是不得已而为之的做法，针对工程应用的数值模拟一般采取这种做法(Delft3D,
CSTM)，理论研究需要应用两相流模型(Hsu et al., 2012; Li and Zhong,
2016)，但计算量太大，而难以实际应用。

床面表层的物质平衡方程为：

![](./media/image2.emf){width="3.0416666666666665in"
height="0.513999343832021in"}

式中，*m~n~*为表层(*k*=1/2)的第*n*分组的泥沙质量，淤积率*D*与第*n*组的悬移质泥沙浓度*c~n~*有关，冲刷率*E~n~*与床面剪切力有关，推移质输沙率*q*和*F~f2~*代表与第2层床沙的交换通量。

悬移质泥沙交换通量仅在一个网格单元计算，而推移质交换通量依赖于相邻的网格单元。

非粘性沙推移质输沙计算公式有很多(Garcia,
2006)。非粘性沙悬移质冲刷率一般根据近床面含沙量计算(Garcia and Parker,
1991)，近床面含沙浓度与床面剪切力、临界床面剪切力(Shields曲线)和粒径有关。当泥沙颗粒之间存在相互作用时（如隐蔽与暴露、粘性力等），准确计算冲刷率很困难，一般采用下式的经验公式计算：

![](./media/image3.wmf)

式中，*r*=1，以及经验参数*M*和![](./media/image4.wmf)。

## 4 Eulerian-Lagrangian framework

河床分为固定数目的分层(*N~lay~*)，每层含有一定分组数的泥沙(*N~frac~*)。

只有河床表层厚度由于冲淤发生变化，以下分层(2:*N~lay~*)厚度不发生直接性变化。分层数对计算耗时将没有影响，仅当河床发生冲刷时及掺混（如生物扰动和床面形态迁移）时变得很重要。

![](./media/image5.emf){width="3.5972222222222223in"
height="2.3406583552055995in"}

Figure 2.1: Sketch of the framework framework with definitions of fluxes
and indices

每层中每组泥沙的质量定义为*M~i,j~*, *i*代表分层，*j*代表泥沙分组。

分层厚度![](./media/image6.wmf)和质量百分比![](./media/image7.wmf)：

![](./media/image8.emf){width="1.947257217847769in"
height="1.2322944006999126in"}

式中，![](./media/image9.wmf)代表泥沙体积百分比。![](./media/image9.wmf)与固结模型耦合时才会变化，其他情况不变。

顶层的泥沙质量平衡方程为：

![](./media/image10.emf){width="3.4097222222222223in"
height="0.4810148731408574in"}

连续冲刷顶层厚度减小，为防止表层泥沙质量变为负值，表层将于第2层合并；连续淤积时，为防止厚度过大，表层将分裂为2层。

河床高程变化定义为：

![](./media/image11.emf){width="3.4027777777777777in"
height="0.6221905074365705in"}

一般表层厚度要比下面分层厚度要薄，表层以![](./media/image12.wmf)速率向下移动。表层泥沙质量变化如下：

![](./media/image13.emf){width="5.413632983377078in"
height="0.5413637357830271in"}

式中，下标![](./media/image14.wmf)为第1层与第2层交界面处的质量比例。当表层厚度不变时，第2层厚度将发生变化。第2层的质量平衡方程变为：

![](./media/image15.emf){width="3.6666666666666665in"
height="0.6386384514435696in"}

以上计算方法可扩展到所有分层，当所有分层厚度不变时，即Lagrangian模式。

（1）所有分层厚度不变时的泥沙质量平衡方程可写作：

![](./media/image16.emf){width="3.513888888888889in"
height="0.592833552055993in"}

（2）分层厚度变化时的泥沙质量平衡方程可写作：

![](./media/image17.emf){width="3.5069444444444446in"
height="0.6522058180227471in"}

式中，![](./media/image18.wmf)为厚度变化分层的标记。通过选择*iv*设置床沙计算模式。![](./media/image19.wmf)为Euleriran模式，![](./media/image20.wmf)为Lagrangian模式。

## 5 Spatial discretization

In case of multiple layers with constant thickness, the first order
upwind scheme will result in numerical diffusion. In case of
stratification, the bed will be mixed due to the up and down movement of
the grid. A higher order scheme might be desirable. Note that Van Ledden
\[2003\] used a central difference scheme in combination with
the*θ*-method for time-stepping. This is only possible in case of two
fractions, as only one fraction has to be resolved in combination with
mass conservation.

## 6 Time-stepping

## 7 Treatment of splitting and merging of the bed cells

In case of degradation, the thickness of the variable layer decreases
and might become negative. To circumvent a negative layer thickness, the
variable layer is merged with the lower layer when a certain thickness
is reached, e.g. 10% of the original thickness. The old lower layer
becomes variable layer and all underlying layers move one layer upwards.
A similar procedure is followed for continuous aggradation. Then, the
variable layer is split and all underlying layers move one layer
downwards. To keep the same number of layers, a layer is added at the
bottom in case of degradation and removed in case of aggradation. There
are different ways to deal with the lower layer: (1) The lower layer is
split or merged with the layer above; (2) a new layer is added or the
lower layer is removed. We prefer the second option as for the first
option a deformed grid is generated in case of severe erosion.
Furthermore, degradation can not go further than the predefined depth. A
drawback of the second method is that in case of high aggradation, the
information of the lower stratification is lost. Note however that this
is the case for the first option as well as the lower layers are merged.

## 8 Mixing in the bed

以上仅考虑了河床冲淤时发生的床沙级配调整。事实上，由于物理作用或生物扰等等掺混过程，分层间会发生相互作用。原则上，掺混过程是非扩散的，但Delft3D模型以扩散型相互作用来考虑分层间的混合。中心差分的离散方程如下：

![](./media/image21.emf){width="3.3055555555555554in"
height="0.9765135608048994in"}

## 参考文献

Urs Neumeier, Christian Ferrarin, Carl L. Amos, Georg Umgiesser, and
Michael Z. Li. 2008. Sedtrans05: An improved sediment-transport model
for continental shelves and coastal waters with a new algorithm for
cohesive sediments. Computers & Geosciences, 34(10): 1223-1242.

M Van Ledden, ZB Wang, H Winterwerp, and H de Vriend. 2004. Sand-mud
morphodynamics in a short tidal basin. Ocean Dynamics, 54(3-4):385-391.

Lawrence P. Sanford. 2008. Modeling a dynamically varying mixed sediment
bed with erosion, deposition, bioturbation, consolidation, and armoring.
Computers & Geosciences, 34 (10):1263-1283.
