求解Exner方程，解决地形更新的数值不稳定性

Fortunato, A. B., and A. Oliveira (2007b), Improving the stability of a
morphodynamic modeling system, Journal of Coastal Research, 50, 486-490.

SAND2D by Fortunato(2004)

Fortunato, A. B., and A. Oliveira (2004), A modeling system for tidally
driven long-term morphodynamics, Journal of Hydraulic Research, 42 (4),
426--434.

与ADCIRC水动力模型耦合计算，平面二维FEM模型。

主要介绍了抑制求解Exner方程引起的地形演变震荡的问题的方法：

(1)a weak non-linear filter is applied at every morphodynamic time step
to eliminate local extremes in the bathymetry

\(2\) use FVM to solve transport equation

\(3\) finer mesh for transport and bathymetry, and coarser grid for flow
simulation

临界坡度调整方法

Roelvink, D., A. Reniers, A. van Dongeren, J. van Thiel de Vries, R.
McCall, and J. Lescinski (2009), Modelling storm impacts on beaches,
dunes and barrier islands, Coastal Engineering, 56 (11-12), 1133-1152.

更新地形时考虑坡度超过临界坡度时发生崩塌(avalanching)的情况：

![](./media/image1.emf){width="1.025in" height="0.43333333333333335in"}

在*y*方向的表达式相同。对干地形和湿地形考虑不同的临界坡度，分别是1.0和0.3。当某一个计算节点的水下坡度超过临界坡度时，下一时刻即调整相邻两个节点间的地形直到达到临界坡度，在随后的计算步将发生连锁反应，此时干地形的节点坡度又会超过临界坡度，最后的结果就是泥沙从干地形向湿地形输移。

上述过程可以通过求解浮泥运动的微分方程实现（机理模拟），但上述的简化模型可方便应用于2D和3D河床演变模型。
