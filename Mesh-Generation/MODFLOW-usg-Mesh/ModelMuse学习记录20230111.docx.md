# ModelMuse Version 4: A Graphical User Interface for MODFLOW 6

## 摘要

ModelMuse支持MODFLOW6，支持两类空间离散：结构网格（DIS）和节点离散（DISV）。使用四叉树网格从一个结构网格模型生成DISV模型。ModelMuse使用新算法的GNC改善DISV模型。ModelMuse不支持其他类型的DISV网格和非结构网格。

ModelMuse支持将单个单元定义为承压或可转换单元，以及删除与不连续层有关的未激活单元，降低计算量。

ModelMuse支持全3D，空间变化的水力传导度。

ModelMuse提供向后兼容。可自动将一些之前版本的软件输入文件转换为MODFLOW6的软件输入，除了SFR。

ModelMuse支持MODPATH和ZONEBUDGET的版本，与MODFLOW6兼容。

## 使用DISV网格离散

ModelMuse4仅支持DISV，即quadtree-refined网格。

MODPATH7与四叉树网格的DISV网格模型兼容（无旋转，带结构网格）。

ModelMuse4生成四叉树加密网格的步骤：

（1）设置Model类型为MODFLOW6

（2）如果尚未创建网格，生成结构网格

（3）（选择性的）指定加密级数

（4）从结构网格切换至DISV网格，或刷新DISV网格

注意GUI界面的按钮图标。

图2：如何定义四叉树网格分级数的截图

图3：显示DISV离散的菜单和工具条

## 带对象的数据定义

用户定义z公式的数字(zero, one,
two)，是否为相交的或封闭的单元或通过插值设置数据，为模型特征的数据集或属性定义公式。

## GNC软件包

默认ModelMuse中为DISV网格打开GNC软件。

ModelMuse使用的GNC算法与GRIDGEN相似，但不同，见图4. 在GRIDGEN中，ghost
nodes的位置不总是落在连接对ghost-node水头有贡献的单元连线上，而ModelMuse对gnc单元有贡献包含在对偏移的修正。见图5.

## MODFLOW6中的可转换单元

MODFLOW6不再设置分层是承压层或可转换层。而是各单元使用ICONVERT数据集（Storage软件中）定义，该数据集对应ModelMuse中的Convertiable数据集。

当Convertiable设为TRUE，一个单元可以在承压层和可转换层之间切换，否则。一个单元永远是承压层。

MODFLOW6的相关数据集是ICELLTYPE（NPF软件包），对应于MOdelMuse中的Cell_Type数据集。

在Cell_Type数据集中定义如下值，指定单元厚度如何计算：

![](./media/image1.emf){width="3.7788331146106735in"
height="1.031350612423447in"}

## XT3D选项

## 模拟不连续分层

DIS和DISV单元可设置为\"vertical pass-through
cells\"，这些单元从模拟从排除，通过IDOMAIN数组的值识别（负值）。IDOMAIN设为0是inactive单元，正数为active单元。

## 模型特征

## 后处理
