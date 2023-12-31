# defmod模型介绍

## 特点

-   Defmod - Parallel multiphysics finite element code for modeling
    > crustal deformation during the earthquake/rifting cycle

-   模拟地震和断裂作用周期内地壳变形的并行多物理过程的有限元模型

-   PETSc (Portable, Extensible Toolkit for Scientific computation）

## 1、Defmod模型简介

Defmod模型，完全非结构化网格，2D/3D，并行有限元模型，可模拟时间尺度从毫秒(milliseconds)到数千年的地壳变形。Defmod模型可模拟各向异性介质中，地震或断裂作用周内主要过程引起的地壳变形。具体地，Defmod模型可用于模拟由于动态或准静态过程，诸如同震破裂位移(co-seismic
slip)、岩脉入侵(dike
intrusions)、由于流体运动和震后或粘弹性松弛后的裂后阶段(post-rifting)引起的孔隙弹性回弹(poroelastic
rebound)，引起的地壳变形。

Defmod模型也可用于模拟由于冰川回弹(post-glacier
rebound)、水文荷载/卸载(hydrological
(un)loading)、地下水库的注水或抽水(injection or withdrawal of fluids
from subsurface reservoir)等，等引起的地壳变形。

Defmod模型采用FORTRAN95语言编程，使用PETSc并行稀疏数据结构和隐式求解器；使用三角形、四边形、四面体或八面体网格；可用于共享内存或分布式内存并行机器上，可使用数百到数千个处理器核；支持预设荷载；计算结果保存为ASCII
VTK格式，可用ParaView或VisIT可视化。

下载地址：https://bitbucket.org/stali/defmod/

## 2、引言

数学模型是研究由于地震、火山入侵、水文荷载和人类活动引起的地壳变形的重要手段。通过模拟地壳变形，结合大地测量和地震观测，我们可以深入了解地球物理过程和评估相关参数。基于有限单元法的非结构网格模型，增强了描述材料属性和几何形状分布的灵活性。

Defmod模型基于PETSc库开发，FORTRAN代码仅约有2000行，易于修改和拓展。

## 3、控制方程

Defmod模型的控制方程是求解Cauch运动方程：

![](./media/image1.png) (1)

式中，![](./media/image2.png)为应力张量，*f*为体积力，![](./media/image3.png)为密度，*u*为位移场，![](./media/image4.png)为控制体。

应用有限元法，方程可写作半离散形式 (Zienkiewicz and Taylor, 2000)：

![](./media/image5.png) (2)

其中，*M*为质量矩阵，*C*为衰减矩阵，*K*为刚度矩阵。

Defmod模型应用线性约束方程施加断层滑动/断层开裂和位移/速度边界条件。约束条件使用Lagrange乘子施加，引入下列方程系统：

![](./media/image6.png) (3)

![](./media/image7.png) (4)

式中，*G*为约束矩阵，![](./media/image8.png)为施加约束条件所需要的力，*l*为约束条件值（如预设的断层活动或位移）。

对于动力波传播问题，Defmod模型使用显式求解，对加速度![](./media/image9.png)使用中心差分格式做时间积分，对速度![](./media/image10.png)使用向后差分格式。在时间步![](./media/image11.png)的位移由下式计算：

![](./media/image12.png) (5)

质量矩阵*M*假设是块集中的，因此是对角矩阵，容易求逆矩阵。矩阵*C*是"成比例关系"或Rayleigh衰减的，有：

![](./media/image13.png) (6)

式中，![](./media/image14.png)和![](./media/image15.png)为用户设定的衰减系数。

根据CFL限制条件，显格式中的临界计算时间步长![](./media/image16.png)由下式限制：

![](./media/image17.png) (7)

式中，*L*为最小单元边长，*E*为杨氏模量，![](./media/image3.png)为单元密度。

对于约束条件，Defmod模型使用"向前增量Lagrange乘子方法"(Carpenter et al.,
2005)。为了吸收边界的反射波，需要使用Kuhlemeyer
(1969)建议的局部单元层格式。吸收算法由置于边界节点上法向和切向的系列缓冲器组成，可以完全吸收以法向入射角度接近边界的波，但对于斜角入射或消散波，能量无法全部吸收。

### 3.1 准静态问题

对于准静态的粘弹性问题，式(3)和式(4)中的惯性项可忽略，简化为隐式不定方程组，可使用并行、稀疏矩阵直接求解法或预处理迭代求解法求解（在时间推进计算步内定义）。

在准静态孔隙弹性问题中，在动量方程之外，还得求解连续方程。这导致一个耦合的方程组，形式如下
(Zienkiewicz and Taylor, 2000)：

![](./media/image18.png) (8)

![](./media/image19.png) (9)

式中，*K~e~*和*K~c~*为固体和液体的刚度矩阵，*H*为耦合矩阵，*S*为压缩度矩阵，*p*为压力矢量，*q*为进出流量。

非恒定方程组使用增量荷载格式求解(Smith and Griffiths,
2005)。为解决对线性单元的Ladyzenskaja-Babuska-Brezzi限制，Defmod模型使用Bochev
and Dohrmann
(2006)建议的局部压力投影格式计算，该格式对线性四边形和六面体单元很有效
(White and Borjam
2008)，但在关于孔隙弹性问题文献中使用线性三角形和四面体单元的有效性未作评述。只要使用高阶的积分格式，如三角形单元的3点积分或四面体的4点积分，稳定性就很好。例如，图1显示的2维孔隙弹性区域，离散使用线性三角形单元，模拟的在一个逆冲断层上的同震滑动之后的压力场（稳定和不稳定计算）。

![](./media/image20.png)

图1
使用线性三角形单元模拟的逆冲断层上同震滑动后2D孔隙弹性区域内的压力场（左图：施加稳定性；右图：未施加稳定性）

### 3.2 动态的粘弹性问题

粘弹性松弛的隐式时间分步算法是基于Melosh and Raefsky (1980)。

## 4、并行化

Defmod模型通过区域分区(domain
decomposition)并行化，一个计算节点(rank)分配一组单元和节点，以及一组全局稀疏矩阵和矢量(PETSc分区的行方向)。所有单元层计算，如局部矩阵构造
(*K*和*M*）、应力恢复等，分配到各计算节点。在分布式矩阵和矢量上的集合和线性代数计算，如稀疏矩阵-矢量相乘，需要使用有效的网格分区和节点重编号的MPI通信，各计算节点间的通信量应最小化。图2显示了使用METIS
(Karypis and Kumar,
1998)和4个计算核心的分区网格。相应的刚度矩阵*K*如图3所示。需要MPI通信在每计算步内更新\"ghost\"节点上的计算值。

![](./media/image21.png)

图2 四个计算核心时使用线性四面体单元的3D网格及分区

![](./media/image22.png)

图3 斜对角上非零系数的稀疏性降低了矩阵-矢量相乘引起的MPI通信量

## 5、程序编译和运行

### 5.1第三方库

使用Cygwin编译了Win32的PETSc_v3.7库，可供Visual Studio 2008调用。

使用METIS5.0库用于网格分区。

### 5.2输入文件

Defmod模型使用一个ASCII格式的输入文件，其中包含所有的模拟参数、网格信息等，诸如：节点坐标和单元连接关系数据、材料数据、非恒定线性限制方程以及非恒定力/流量或者牵引/通量边界条件。

输入文件还包括：固定的或滚轴式边界信息、为模拟地壳均衡效用的Winkler地基以及吸收边界。

为生成网格，如节点坐标和连接性数据，需要使用Gmsh或Cubit网格划分软件。

### 5.3运行

mpiexec -n 2 defmod -f two_quads_qs.inp

因为Defmod模型使用PETSc，因此所有PETSc命令行选项都可以在Defmod中使用。可使用-help开关获得选项列表。默认地，准静态问题使用由GMRES
(generalized Minimal REsidual)方法预处理的ASM (Additive Schwarz
Method)求解。对于无约束问题，通过设定命令行选项 -ksp_type
cg使用共轭梯度法(CG)求解。

如果PETSc安装配置了并行稀疏直接求解器，如MUMPS (MUltifrontal Massively
Parallel Solver)，则可以使用命令行选:

-pc_type lu -pc_factor_mat_solver_package mumps

使用该功能。

用户应尝试使用PETSc中的各种求解器。

### 5.4 施加边界条件

可使用在断层两侧的重合节点模拟断层滑动，如空间上2个节点的坐标相同，但属于不同单元。为施加2个重合节点间的滑动，必须使用线性限制方程。例如，图4所示的在节点3和节点8之间定义5m的开口，分别属于单元1和单元2，需要使用2个方程：![](./media/image23.png)和![](./media/image24.png)，该方程可以在输入文件中定义，在时间推进计算过程中激活限制条件。

限制方程还可以用于制造一个可渗透的断层，如方程P3-P8=0.0，可保证穿过节点3和节点8的压力连续性。

相似地，方程也可用于施加非恒定的非零位移和(或)不同荷载方案(有无断层)下的压力边界条件。

![](./media/image25.png)

图4 断层界面显示重合节点对({3,8}和{2,5})来描述断层滑动/张开

### 5.5 输出及可视化

默认情况下，每个计算节点(rank)输出自己的ASCII
VTK格式的输出文件，可以使用ParaView，VisIT等软件可视化。

用户也可以使用标准的Linux shell应用程序处理输出文件。

图S1显示了使用Defmod模型获得的一些计算结果可视化。

## 6、验证

与Abaqus软件计算对比；网格使用Cubit生成；所有例子中，杨氏模量*E*=30.0
GPa，泊松比为0.25

\[1\]弹性动力学算例，Lamb问题变体，计算一个100km
![](./media/image26.png)50km的2D弹性区域，使用线性四边形单元离散，在自由表面施加一个瞬时的点力(垂向上-10.0GN)。计算表面距离点源25km处的位移。假设刚度比例衰减系数![](./media/image27.png)。图5显示了Defmod计算的250步厚的波场。图6显示的是Defmod和Abaqus计算的随时间变化的垂向和水平位移。

![](./media/image28.png)

图5 在250时间步时的位移变量(![](./media/image29.png))-波场

![](./media/image30.png)

图6 距冲击源25km的表面处水平(X轴)和垂向(Y轴)位移

\[2\]准静态孔隙弹性算例，Terzaghi固结问题，模拟由于突然施加荷载分布后孔隙压力变化。区域范围：长度1km![](./media/image26.png)宽度1km。底部边界节点固定，其他节点允许垂向位移。顶部表面有流体流过，其上施加1.0MPa的突然荷载。假设整个区域的渗透系数(水力传导度)为10^-9^
m/s。图7显示的是1s后孔隙压力分布。图8显示的是1s后沿着垂直剖面Defmod和Abaqus的计算结果。Defmod模型使用稳定化的线性单元，因此可以较Abaqus更快速地计算压力场。

![](./media/image31.png)

图7 Terzaghi固结问题在1s时的空隙压力场

![](./media/image32.png)

图8 在*t*=1*s*时随深度变化的孔隙压力变化

\[3\]准静态粘弹性算例，在一个100km长的平移断层(strike-slip
fault)上发生1m的预设滑动，从表面延伸至25km深度。假设弹性地壳厚25km，其下是225km厚的粘弹性地幔，地幔的Maxwell粘度为![](./media/image33.png)=10^18^
Pa-s。模拟区域：500km![](./media/image26.png)500km![](./media/image26.png)250km，使用3D线性六面体单元离散，部分区域显示于图9。Defmod和Abaqus在t=0和t=10年时，沿着垂直于断层的断面，表面计算结果显示于图10。

![](./media/image34.png)

图9
用于验证准静态粘弹性松弛模型的部分有限元网格，颜色表示*t*=年时断层平行位移

![](./media/image35.png)

图10 在X方向断层平行表面位移
