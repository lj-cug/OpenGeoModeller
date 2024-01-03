# CUDA各版本之间有什么区别

## 摘要

CUDA是英伟达的GPU通用计算（GPGPU，General Purpose comuputing on
GPU)架构。不同版本的CUDA的区别主要在"GP"和"GPU"上：

1.  支持的计算库多少和计算销量（GP）

2.  支持的GPU架构新旧（GPU）

当然还有一些对语言版本的支持，编译器的优化（CUDA9开始支持c++14，gcc6），bug
fixes等等。

具体区别

以现在比较常见的CUDA8.0，9.x，10.x为例，从计算库和支持的计算架构两个方面来罗列一下他们的区别：

**计算库：**

-   CUDA8.0支持的lib/中有很常用的：cublas（基础线性代数计算库，Basic
    Linear Algebra
    Subprograms）,cufft（快速傅里叶变换）,curand（随机数）,cusparse（稀疏矩阵），cusolver（解线性方程组）等等。

-   CUDA9.0在cusolver，cugraph库中加入了新算法；加速了cublas，cufft中的算法；引入了Cooperative
    Groups来优化线程同步问题。

-   CUDA9.2对RNN和CNN都做了一些专属加速；给cuFFT再次加速；加入了cutlass（自定义的线性代数运算）。

-   CUDA10.0加入了nvJPEG（jpg图像处理库），应当可以加速dataloader；优化了cublas和cufft的性能。

## 支持的计算架构（Micro-architecture）：

CUDA8.0：

-   费米（Fermi，GTX580）

-   开普勒（Kepler，GTX680，GTX780Ti，GTX Titan，Titan Z，Tesla K80）

-   麦克斯韦（Maxwell，GTX980Ti，Titan X，Tesla M40）

-   帕斯卡（Pascal，GTX1080Ti，Titan Xp，Tesla P100）

CUDA9.x：

-   开普勒（Kepler，GTX680，GTX780Ti，GTX Titan，Titan Z，Tesla K80）

-   麦克斯韦（Maxwell，GTX980Ti，Titan X，Tesla M40）

-   帕斯卡（Pascal，GTX1080Ti，Titan Xp，Tesla P100）

-   伏特（Volta，Titan V，Tesla V100）

CUDA10.x：

-   开普勒（Kepler，GTX680，GTX780Ti，GTX Titan，Titan Z，Tesla K80）

-   麦克斯韦（Maxwell，GTX980Ti，Titan X，Tesla M40）

-   帕斯卡（Pascal，GTX1080Ti，Titan Xp，Tesla P100）

-   伏特（Volta，Titan V，Tesla V100）

-   图灵（Turing，RTX2080Ti，Titan RTX，Tesla T4）

**版本选择**

越新的版本性能就越好吗？

是的。

如何选择版本以最大化GPU的利用效率？

全都装，不同版本的DL框架会用到不同的CUDA版本（tf尤其挑剔）。用的时候再export。
