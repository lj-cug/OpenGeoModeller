# 五、絮凝动力学(Verney et al., 2011)

学习过程（沈洋）：

Andrew J. Manning et al., Flocculation dynamics of mud
：学习关于非粘性沙和粘性沙混合沙的物理-化学-生物特性及研究方法，重点学习其环形水槽的观测结果的分析方法，而不是其试验细节；

横向振动格栅的试验系统及试验方法，重点参考重庆交大的硕士论文，试验需要钟强指导，该部分要掌握试验步骤，图片处理（2个MATLAB程序），特别是试验组次设计（要多交流）；

Verney et al.,
2011一文要细细读，阅读该文的试验方法与模拟方法，特别是FLOCMOD模型的原理及其中考虑的絮凝控制力，通过该文学习，将对絮凝过程有了更深入的了解。我有该文的Fortran程序，后期需要结合试验，研究絮凝团的沉速与大小（Euler）和絮凝团的发展过程与作用力（Lagrange）。

![](./media/image1.emf){width="4.030555555555556in"
height="1.5305555555555554in"}

图5.1 絮凝过程示意图

quantified flocculation processes and changes in floc populations over
time using *in situ* measurements during tidal cycles

the major changes in floc populations during the different stages of the
tidal cycle, and a strong dependency on hydrodynamic conditions, often
quantified by shear rate, *G*, or the Kolmogorov microscale, *η*.

但是，采样都是Euler式的：（1）不能观测絮凝群体(population of
flocs)在周围湍流作用力影响下的行为；（2）仅有观测窗口或采样容器内的颗粒，不能知道之前絮凝群体的历史。

聚合过程(aggregation
processes)非常复杂，持续时间从若干分钟到若干小时，了解絮凝体(floc)的行为与发育历史信息非常必要。

process-based flocculation model：

i\) size-class based models (SCB)

ii\) distribution-based models

iii\) a simplified model based on changes in a characteristic diameter
over time (Winterwerp, 2002)

以上三种模型的优缺点：

Computation of the third model is fast (only one class simulated) but it
only provides limited information on the floc population, and although
suitable for operational issues, it is too simple to investigate the
behaviour of the floc population. The distribution-based model is rapid
to compute and provides more knowledge on floc size distribution but is
limited by the assumption of a fixed distribution: bimodal populations
as observed by Benson and French (2007) could not be computed with such
distribution-based models.

SCB models are the best way to investigate floc behaviour as they
provide detailed information on floc populations. However, the SCB
models described in the literature are limited by the large number of
size classes required to correctly reproduce aggregation/fragmentation
processes and their computation costs are thus high. This explains why
they have not yet been coupled with hydrodynamic and sediment transport
models.

## 5.1 试验装置

A 'video in lab' (VIL) device was used in this study both to observe
floc populations and to artificially reproduce variations in shear rate
at the tidal scale

This device consists of a small (13 cm wide, 20 cm high) cylindrical
test chamber equipped with a ten-speed impeller to control turbulent
agitation. The turbulent field generated inside the test chamber was
calibrated from turbulent kinetic energy measurements performed with a
laser Doppler velocimeter (Verney, 2006). Results confirmed homogeneous
shear rates inside the cylinder for all impeller rotation speeds
(corresponding to equivalent shear rates of from 0 to 12 s^-1^).

The 'video in lab' was equipped with a Sony CCD camera and lens that
provide 8 μm pixel resolution images for the maximum enlargement. A
backlight was placed opposite the CCD camera to provide a uniform white
background upon which flocs appear as grey scale silhouettes. Particle
separation is automatically processed by the ELLIX® software
(Microvision®), which uses image thresholding to detect particles larger
than 50 μm.

## 5.2 Model description (FLOCMOD)

### 5.2.1 Floc characteristics and size distribution

FLOCMOD is the size-class-based (SCB) model used to reproduce the
flocculation and fragmentation processes.

the fractal behaviour of flocs was assumed to be the main floc
characteristic sizes in each class *i* (diameter Di, mass mi and density
ρf,i) and can be expressed via the fractal dimension nf:

![](./media/image2.emf){width="2.4335498687664043in"
height="1.4969805336832895in"}

The number of size classes required for SCB models depends on their
structure. Three methods are available. （泥沙颗粒级配组数目确定方法）

A first group of models (Lagrangian) uses as many classes as created
particles (Maggi et al., 2007). This can yield hundreds of classes and
may lead to high computation costs when integrated in 2D/3D models.

A second group uses size classes defined by mean sizes (or mean masses)
and lower and upper boundaries.

In order to optimize the number of size classes required to
realistically reproduce flocculation processes, a mass interpolation
scheme is applied in FLOCMOD (Xu et al., 2008) for aggregation and
fragmentation. Each class corresponds to a typical floc size,
logarithmically distributed from the primary particle diameter *Dp* to
the maximum floc size *Dmax* and boundaries between these classes are
ignored.

### 5.2.2 Flocculation processes

Particle exchanges between classes are allowed through processes limited
to two-body interactions (McAnally and Mehta, 2001) and governed by
aggregation, shear breakup and collision breakup terms (both for gain
and loss of particles) as described hereafter in the generic equation,
where *nk* is the number of particles in the k class (in m^-3^):

每组絮凝团之间的颗粒交换只限于两两作用，受聚合力、剪切破碎力和碰撞破碎力控制(McAnally
and Mehta,
2001)，导致各组絮凝泥沙颗粒的增加或减小，*n~k~*表示*k*组絮凝团中的泥沙颗粒数目(m^-3^):

![](./media/image3.emf){width="5.768055555555556in"
height="0.3968558617672791in"}

\(1\) Aggregation (Gaggr and Laggr)

These terms correspond to the gain or loss of class k particles when i)
two particles collide and i i) the collision is efficient, i.e. the
newly formed bound between the two particles can withstand the shear
induced by the collision.

In the present study (0D), shear aggregation is assumed to be dominant
and differential settling aggregation is neglected. Therefore, the
two-body collision probability function *A(i,j)* is a function of the
shear rate G and particle diameters *Di* and *Dj*

![](./media/image4.emf){width="2.4185608048993874in"
height="1.7444586614173228in"}

碰撞系数![](./media/image5.wmf)表征泥沙颗粒粘性度，如物理化学力和有机物的粘着力。研究者已开展了很多碰撞系数![](./media/image5.wmf)数学描述和计算公式的研究(Maggi
et al.,
2007)，但对粘性沙的物理-化学-生物特性的了解还是知之甚少。FLOCMOD模型中![](./media/image5.wmf)为常数取值（与絮凝团尺寸无关），与破碎速率(fragmentation
rate)![](./media/image6.wmf)作为优化参数使用。

\(2\) Shear breakup (Gbreak_shear and Lbreak_shear)

These terms represent the gain and loss of *k*-class particles from the
fragmentation of larger flocs (*i*\>=*k*) induced by the turbulent shear
stress. The probability of a *k*-class floc to break up for a given *G*
value is expressed by the fragmentation probability function Bi as
proposed by (Winterwerp, 2002).

![](./media/image7.emf){width="2.9255719597550307in"
height="1.1009776902887138in"}

![](./media/image8.emf){width="2.348141951006124in"
height="0.7547408136482939in"}

式中，![](./media/image9.wmf)为絮凝团的破碎速率，为絮凝团屈服应力的函数。与碰撞系数![](./media/image5.wmf)类似，![](./media/image9.wmf)与絮凝团的物理化学特性和有机质含量有关。模型计算中，![](./media/image9.wmf)也取常数值，作为优化参数。

李健：关于絮凝团的碰撞与破碎过程，有试验观测(Shen,
2016)、振动格栅试验(Alan et al.,
2010)和直接数值模拟、DEM的理论探讨(Zhang et al.,
2016)，研究成果是否对以上取值有启发？

根据Winterwerp的絮凝团平衡尺寸与Kolmogorov微观尺度的假定，![](./media/image10.wmf)取定值3/2，![](./media/image11.wmf)(Winterwerp,
2002)。

FDBSij
为剪切破碎后的碎片絮凝团的分布函数，对此的了解还是知之甚少，仅能参考Maggi(2005)的理论探讨。为探讨分布函数的不确定性对絮凝群体动力学计算结果的影响，可选择使用以下三种分布函数：

i\) Binary distribution, i.e. the fragmentation of a mi-mass particle in
two particles of equivalent mass mi/2

"一分二"式的分布函数：一个m~i~质量的颗粒分裂为2个同等质量mi/2的颗粒。

![](./media/image12.emf){width="2.2290244969378827in"
height="0.7744411636045494in"}

ii\) Ternary distribution, i.e. the fragmentation of a mi-mass particle
in one particle of mass mi/2 and two particles of equivalent mass mi/4

"一分三"式的分布函数：一个m~i~质量的颗粒分裂为1个质量mi/2的颗粒和2个mi/4的颗粒。

![](./media/image13.emf){width="2.5014741907261593in"
height="1.5544433508311462in"}

iii\) Erosion, i.e. the fragmentation of a mi-mass particle in one large
fragment and several (*k*) small fragments of the same mass mk (Hill,
1996).

"一分多"式的分布函数：一个m~i~质量的颗粒分裂为*k*个相同质量mk的颗粒。

![](./media/image14.emf){width="3.494504593175853in"
height="1.1010258092738407in"}

\(3\) Collision-induced break-up (Gbreak_coll and Lbreak_coll)

Two-floc collisions can lead to aggregation but also to break-up if the
ollision-induced shear stress (τ*coll* ) experienced by the particles
exceeds the strength (τ*y*) of at least one of the two colliding
particles. Gain and loss terms of collision-induced break-up
(Gbreak_coll and Lbreak_coll) are defined as proposed by McAnally and
Mehta (2001) similarly to the shear aggregation terms Gaggr(k) and
Laggr(k):

![](./media/image15.emf){width="3.4038320209973754in"
height="1.276654636920385in"}

式中，*A(i,j)*为絮凝团*i*和絮凝团*j*之间的碰撞概率分布函数。

由碰撞引起的破碎分布函数![](./media/image16.wmf)决定：（1）发生碰撞破碎：颗粒所有的碰撞引发的剪切力![](./media/image17.wmf)大于其中一个碰撞颗粒的屈服应力![](./media/image18.wmf)；（2）絮凝碎片分布如何。计算![](./media/image19.wmf)需要首先计算![](./media/image17.wmf)和![](./media/image18.wmf)。

碰撞引起的剪切力![](./media/image17.wmf)由下式计算：

![](./media/image20.emf){width="3.032726377952756in"
height="1.147250656167979in"}

上式中絮凝团*i*的屈服应力(![](./media/image21.wmf))可采用絮凝团密度（与絮凝团尺寸有关）计算：

![](./media/image22.emf){width="1.964410542432196in"
height="0.8671609798775153in"}

*F~p~* represents the relative depth of inter-particle penetration and
is estimated to be 0.1(McAnally and Mehta, 2001). *F~y~* represents the
yield strength. Estimation of this parameter is highly empirical and a
constant value of O{10^-10^} N is used hereafter as proposed by
(Winterwerp, 2002).

对于两个絮凝团的相互作用，可能会发生2种类型的碰撞，这决定了碰撞引起的破碎分布函数![](./media/image19.wmf)的计算方法(McAnally
and Mehta, 2001):

\(1\) ![](./media/image23.wmf) and ![](./media/image24.wmf):

The collision-induced shear stress exceeds the shear strength of the
weakest aggregate only, consequently, during the collision, the j
particle breaks into two fragments (Fj,1 and Fj,2) such as mFj,1=13/16mj
and mFj,2=3/16mj. Fj,1 is a free fragment while Fj,2 is bound with the i
particle.

其中有一个絮凝团的碰撞剪切力大于其屈服应力，发生破碎，破碎后分为2个絮凝团，并附着于未破碎的絮凝团上。最后还是2个絮凝团！

\(2\) ![](./media/image25.wmf) and ![](./media/image24.wmf):

The shear stress is larger than shear strength of both i and j
particles. Both particles break into two fragments (Fi,1 ; Fi,2) and
(Fj,1 ; Fj,2) such as \[mFi,1; mFj,1\]=13/16 \[mj ; mj\] and \[mFi,2;
mFj,2\]= 3/16 \[mj ; mj\]. Two particles are formed from the two primary
particles Fi,1 and Fj,1. A third particle is formed from the two
fragments (Fi,2 and Fj,2) that bound during collision.

两个絮凝团的碰撞剪切力都大于各自的屈服应力，都分裂为2个絮凝团（共有4个絮凝团了！），其中包括2个主要尺寸的絮凝团，剩下的2个碎片附着形成1个絮凝团（共3个絮凝团！）。可以认为：不管1个絮凝团破碎为几个，最终都是2个主要尺寸的絮凝团存在，剩下的碎片全部附着在一起形成1个新的絮凝团。

### 5.2.3 Time step

An explicit integration scheme was adopted to solve the aggregation and
fragmentation equations such as Eq. 6 can be written as:

![](./media/image26.emf){width="5.768055555555556in"
height="0.5278390201224847in"}

Aggregation and fragmentation occur simultaneously for all size classes
and careful attention must be paid to the time step used to ensure mass
conservation and that all classes do not receive a negative number of
particles. In this case, two options are available:

i\) a constant time step, which must be small, i.e. equal or lower than
1 s;

ii\) a varying time step, which prevents more particles aggregating or
fragmenting than exist in each class;

## 参考文献

1.  Alan Cuthbertson, Dong Ping, Peter Davies.
    [Non-equilibrium flocculation characteristics of
    fine-grained sediments in grid-generated turbulent
    flow](http://www.sciencedirect.com/science/article/pii/S0378383909001902).
    Coastal Engineering, 2010,57(4): 447-460. （振动格栅试验）

2.  Shen Xiaoteng, Jerome P.-Y. Maa. A camera and image processing
    system for floc size distributions of suspended particles. Marine
    Geology 376 (2016) 132-146.

3.  Shen Xiaoteng, Jerome P.-Y. Maa. Numerical simulations of particle
    size distributions: Comparison with analytical solutions and
    kaolinite flocculation experiments. Marine Geology 379 (2016) 84-99.

4.  Verney, R., Lafite, R., Claude Brun-Cottan, J., & Le Hir, P.
    (2011).Behaviour of a floc population during a tidal cycle:
    laboratory experiments and numerical modelling. Continental Shelf
    Research, 31(10), S64-S83.

5.  Xu Fanghua, Dong-Ping Wang, Nicole Riemer. Modeling flocculation
    processes of fine-grained particles using a size resolved method:
    Comparison with published laboratory experiments. Continental Shelf
    Research 28 (2008) 2668-2677.

6.  Zhang Jin-Feng, Jerome P.-Y. Maa, Zhang Qing-He, et al., 2011.
    Direct numerical simulations of collision efficiency of cohesive
    sediments. Estuarine, Coastal and Shelf Science 178: 92-100.
