# MODFLOW6

MODFLOW软件有：

(1)MODFLOW-2005 (笛卡尔网格）

(2)MODFLOW-USG (非结构网格）

(3)MODFLOW6(面向对象编程，统一之前版本的MODFLOW）

(4)MODFLOW-parallel(Delft三角洲研究院开发的MPI并行版本MODFLOW6）。

MODPATH7：基于MODFLOW的地下水粒子轨迹跟踪模型。

MODFLOW前处理：Flopy, SFRmaker等Python脚本程序、GRIDGEN (生成quadtree网格）

MODFLOW后处理：Flopy, ModelMuse, mvmf6-1.0.0

MODFLOW-setup：USGS开发的自动化建模的Python程序（限于结构网格）

iMOD: Delft开发的MODFLOW建模的Python程序（针对并行化的MODFLOW）

Wisconsin_Central_sands：基于MODFLOW6的地下水建模项目备份

因此，MODFLOW的基本学习路线应该是：

参考Pleasant Lake的tutorial，针对塔里木盆地的地下水运动建立MODFLOW模型。

前处理：ArcGIS处理一些栅格和矢量数据，基础数据导入Flopy，建立MODFLOW6模型。

运行MODFLOW6：编译源码，运行MODFLOW6

后处理：可以使用Flopy，也可以使用USGS开发的后处理程序，如mvmf6-1.0.0

MODFLOW2005是最简单的结构网格版本的MODFLOW，MODFLOW-USG和MODFLOW6的非结构网格版本，较为复杂。需要循序渐进的学习。

在塔里木盆地地下水建模过程，通过实战学习MODLFOW的操作流程，其他区域的地下水建模就可以如法炮制。

MODFLOW的所有版本的源码和tutorial都可以在github上下载到。

MPI并行版本的MODFLOW6有2个版本：

1. MODFLOW6-Verkaik

荷兰Delft研究院Verkaik在博士论文中开发的并行版MODFLOW6，实现了GWF-GWF多模型的粗粒度并行，限制是：仅实现了GWF-GWF的耦合，不能并行计算GWT-GWT模型

2. MODFLOW6-USGS

USGS开发的基于PETSc库的并行版本，即加速求解器的并行，实现了GWF-GWF, GWF-GWT, GWT-GWT的交换并行计算

使用MESON工具编译

编程语言：FORTRAN

快速建模工具Python脚本： modflow-setup以及flopy

# modflow-2005

结构网格版本的MODFLOW

# modflow-usg

非结构网格版本的MODFLOW

modflow-2005与usg版本都已经统一到MODFLOW6的框架下了
