# JTK

Java Toolkit, Dave Hales提倡的使用JAVA开发的地震解释的基础程序库

在JTK的基础上，Wu Xinming开发了断层和层位解释的JAVA程序

在Windows系统下运行需要安装Jython

原始的seg-y格式地震勘探数据，需要使用JTK库工具，转换为JAVA下的大端序二进制格式

## ipf

引入了一种能描述相交断层连接关系的数据结构，相比断层面的网格描述，该数据结构更简单，便于计算机之间的交换，便于后续的断层的图像处理。

Wu Xinming, Dave Hale. 2016. 3D seismic image processing for faults. GEOPHYSICS, VOL. 81, NO. 2

## OSV

本文首先介绍使用动态编程算法（Hale, 2013a）拾取最优路径(2D)和最优面(3D)的方法。传统方法受噪声和地层特征影响很大。此外，断层属性图像内的断层特征通常难以连续跟踪。最优面投票法的计算成本与种子点数目有关，与地震数据量无关，并行化计算效率很高（4核电脑上，超过1000个种子点，在1s内完成计算，得到对应的最优投票的面和最终的投票得分图）。

Wu, X., S. Fomel. 2018, Automatic fault interpretation with optimal surface voting: Geophysics, 83(5): O67-O82.

## MHE

根据局部坡度和多重网格相关系数，自动提取地震层位。

Xinming Wu, Sergey Fomel. 2018. Least-squares horizons with local slopes and multigrid correlations

## 基于深度学习的断层自动提取

Xinming Wu, et al. 2019. FaultSeg3D: Using synthetic data sets to train an end-to-end convolutional neural network for 3D seismic fault segmentation

Wu Xinming, et al. 2019. FaultNet3D: Predicting Fault Probabilities, Strikes, and Dips With a Single Convolutional Neural Network. IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, 57(11)