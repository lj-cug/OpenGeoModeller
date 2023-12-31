**ELCIRC_parallel, SELFE_v3.1dc,SCHISM5.3等海洋动力学模型的总结**

数学模型：Accuracy (error), stability, robustness, efficiency

物理过程：Boussinesq假定,
静水压力的动量方程、物质守恒、海水状态方程和参数化的亚格子尺度输移(subgrid-scale
transports)

模分裂(mode-splitting)：将3D场的时间积分分为正压（水深平均）和斜压（剩余部分），方便压力梯度项的计算(Bryan
and Cox, 1969)。

1、从离散方法看：

有限差分法：POM, TRIM, ROMS, NCOM

有限单元法: SEOM, ADCIRC, QUODDY

有限体积法+有限差分法: UnTRIM, ELCIRC, FVCOM

有限体积法+有限单元法: SELFE, SCHISM

2、从网格模式及计算功能看：

ADCIRC, POM, QUODDY, TRIM, ROMS, NCOM:
结构化网格，难以覆盖河口或河流的复杂边界和局部地形；计算时间步长受限（CFL条件），难以模拟长时间物理过程；计算效率低。

UnTRIM, FVCOM, SUNTANS, ELCIRC, SELFE, SCHISM:
非结构化网格，UnTRIM不能计算斜压部分，过于简化的湍流模型\-\--影响垂向分层、掺混以及水-大气之间热交换等。

3、从数值算法来看：

POM, ROMS, FVCOM, ADCIRC, QUODDY,
SEOM采用显格式的分裂模式算法，除了内外模式分裂计算带来计算误差外，计算稳定性限制条件CFL制约计算时间步长。

UnTRIM, SUNTANS, ELCIRC, SELFE,
SCHISM模型，这些模型隐格式计算正压压力梯度项、动量方程的垂向粘性项和连续方程的散度项（这些项对计算稳定性的限制最严格CFL），其他项采用显格式计算。因此，不用采用分裂为内外模式的计算方法。另外，半隐格式离散后的系数矩阵为正定、对称的稀疏矩阵，可以使用高效率算法求解（如雅克比共轭梯度算法）。对流项使用Euler-Lagrange
Method(ELM）离散，缓和CFL条件限制。

ELCIRC:
使用阶段函数形式的形状函数表示水位，该数值方法过于耗散。ELCIRC使用有线差分法离散，对网格正交性要求高。垂向使用Z坐标系统，在河床或海床附近形成阶梯状分层(staircase)，因此不能模拟底部边界层。

SELFE:
使用Galerkin有线单元法，不需要网格正交；使用线性形状函数表示水位；垂向使用SZ混合坐标系统（general
sigma
coordinate）可跟踪地形变化（水深1m\~1000m），可准确模拟河床摩阻和河口盐水入侵过程。

SCHISM:
水平向可使用四边形、三角形和混合网格，提高计算效率；垂向可使用LSC^2^(localized
sigma coordinate with shaved
cells)网格系统，可减少浅水区的垂向分层并减小底部网格坡度，实现1D-2DH-2DV-3D的无缝衔接；ELM计算中特征线根部插值算法改进，可抑制数值虚假震荡（通常采用粘性项或过滤函数来稳定），因此可应用于浅水区(non-eddying
regime)和深水区(eddying
regime)模拟；物质输移方程：upwind格式适于用于浅水区（耗散性强，精度低，但稳定性好），TVD格式适于用于深水区，TVD格式计算量大于upwind格式。SCHISM模型增加了混合upwind和TVD的TVD^2^算法，可通过设置tvd.prop文件制定哪些区域使用upwind，哪些区域使用TVD算法，兼顾计算效率和计算精度。增加了海岸线干湿变化算法和球坐标系统。

SELFE和SCHISM模型可计算非静水（动水）压力（压力-流速耦合）。

表1 SCHISM系列模型的发展总结

+------------+-------------------+---------------+--------------------+
| 模型       | ELCIRC            | SELFE         | SCHISM             |
|            |                   |               |                    |
| 项目       |                   |               |                    |
+============+===================+===============+====================+
| 离散方法   | 有限差分+有限体积 | 有限          | 有限单元+有限体积  |
|            |                   | 单元+有限体积 |                    |
+------------+-------------------+---------------+--------------------+
| 水平网格   | 三角形\           | 三角形网格    | 三角形             |
|            | \四边形\\混合网格 |               | \\四边形\\混合网格 |
+------------+-------------------+---------------+--------------------+
| 垂向坐标   | Z坐标系统         | S             | SZ混               |
|            |                   | Z混合坐标系统 | 合坐标\\LSC^2^坐标 |
+------------+-------------------+---------------+--------------------+
| 动水压力   | 无                | 有            | 有                 |
+------------+-------------------+---------------+--------------------+
| ELM插值    | 线性插值          | 反距离插值    | 反距离             |
|            |                   |               | \\高阶Kriging插值  |
+------------+-------------------+---------------+--------------------+
| 水平粘性项 | 无                | Laplacian粘性 | Bi-harmonic粘性    |
+------------+-------------------+---------------+--------------------+
| 物         | ELM和upwind       | ELM           | ELM\\u             |
| 质输移方程 |                   | \\upwind\\TVD | pwind\\TVD\\TVD^2^ |
+------------+-------------------+---------------+--------------------+
| 干         | 虚拟水深          | 虚拟水深      | 虚拟               |
| 湿边界计算 |                   |               | 水深/岸线跟踪算法  |
+------------+-------------------+---------------+--------------------+

表2 SCHISM模型与FVCOM模型的对比

+----------+-------------+---------------+----------------------------+
| 项目     | FVCOM       | SCHISM        | 说明                       |
+==========+=============+===============+============================+
| 离散方法 | 有限差      | 有限          | 有                         |
|          | 分+有限体积 | 单元+有限体积 | 限差分法对网格正交性要求高 |
+----------+-------------+---------------+----------------------------+
| 水平网格 | 三角形      | 三角形\       | 三角网格会引起spurious     |
|          |             | \四边形\\混合 | inertial mode              |
+----------+-------------+---------------+----------------------------+
| 垂向坐标 | *σ-S*地     | *S-Z*坐标     | *σ*                        |
|          | 形跟踪坐标  | \\LSC^2^坐标  | 和纯S坐标会引起hydrostatic |
|          |             |               | inconsistency              |
+----------+-------------+---------------+----------------------------+
| 计算模式 | 模分裂      | 半隐格式      | 模分裂引起的计算误差较大   |
|          | /半隐格式\* |               |                            |
+----------+-------------+---------------+----------------------------+
| 时间项   | 1阶Eul      | 1             | Coura                      |
|          | er/2阶R-K（ | 阶Euler/5阶R- | nt数限制(*Cu*≥1)，*Cu*数越 |
|          | 向前差分）  | K（逆向跟踪） | 小，数值扩散越大，增大时间 |
|          |             |               | 步长、细化网格或filter函数 |
+----------+-------------+---------------+----------------------------+
| 动量方   | 2阶迎风格式 | ELM           | EL                         |
| 程对流项 |             |               | M降低CFL条件对时间步长的限 |
|          |             |               | 制，但具ELM有质量不守恒的  |
|          |             |               | 缺点，与**插值算法**有关。 |
+----------+-------------+---------------+----------------------------+
| 垂向流速 | 内外模      | 单元体质      | 模分裂，每一内模时间       |
|          | 计算校正法  | 量守恒计算法  | 步内均需要做调整；质量守恒 |
|          |             |               | 法可适应陡坡情况，在水面和 |
|          |             |               | 河床附近的计算误差可忽略。 |
+----------+-------------+---------------+----------------------------+
| 物质     | 2阶迎风格式 | E             | TVD和TVD^2^质量守恒，TV    |
| 输移方程 |             | LM/TVD/TVD^2^ | D^2^适用于深水区和浅水区。 |
|          | MPDATA+FCT  |               | MPDATA适用于河床陡坡情况。 |
+----------+-------------+---------------+----------------------------+

-   FVCOM3.1版本后增加了半隐格式和动水压力计算模块。

-   MPDATA: multidimensional positive definitive advection transport
    algorithm;

-   FCT: flux corrected transport

时间步计算格式：单步法（蛙跳格式、Adams-Bashforth、Forward-Backward等）和多步法（预测-校正格式，如R-K法）。

单步法已基本不用，原因：流速求解2次，水位求解1次，之间的不协调。

时间步长限制条件：

（1）外模重力波限制（动量方程，CFL条件）：![](./media/image1.png)

（2）内模重力波限制（标量方程）：![](./media/image2.png)，![](./media/image3.png)

（3）水平粘性项限制：![](./media/image4.png)

![](./media/image5.png)\--SELFE模型；

FVCOM模型(Smagorinsky涡参数法，1963)：

动量方程中的水平扩散系数：![](./media/image6.png)

物质输移方程中的水平扩散系数：![](./media/image7.png)

垂向扩散系数使用湍流模型计算（Mellor-Yamada, GOTM）

（2）和（3）限制条件比（1）要小很多。（1）限制条件通过半隐格式因子![](./media/image8.png)来缓解。当标量方程使用TVD格式时要减小时间步长。
