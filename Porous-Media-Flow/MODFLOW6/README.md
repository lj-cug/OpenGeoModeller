# MODFLOW6

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



## 参考文献

