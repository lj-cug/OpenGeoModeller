# 华为KML的SOLVER（稀疏线性方程组求解器）

## 1简介

Kunpeng Boostkit 数学库KML 1.4.0下载联接：

https://www.hikunpeng.com/developer/boostkit/library/math

KML_SOLVER是稀疏线性方程组求解器，稀疏矩阵是指大部分矩阵元素为零的矩阵。该求解器包含的求解方法有：

● CG（Conjugate gradient，共轭梯度）

● GCR（Generalized Conjugate Residual，广义共轭残差）

● 现有版本包含单精度和双精度数据类型，不支持复数。

## 2 CG求解器

函数定义（S表示单精度，D表示双精度，?即S或D）

KmlIssCgInit?I

初始化数据结构，并将用户提供的系数矩阵关联到求解系统中。

[C Interface：]{.mark}

int KmlIssCgInitSI(KmlSolverTask \*\*handle, int n, const float \*a,
const int \*ja, const int \*ia);

int KmlIssCgInitDI(KmlSolverTask \*\*handle, int n, const double \*a,
const int \*ja, const int \*ia);

调用举例：

#include \"kml_iss.h\"

KmlSolverTask \*handle

int n = 8;

double a\[17\] = {
1.0,1.0,2.0,9.0,2.0,1.0,-3.0,3.0,2.0,9.0,-5.0,6.0,1.0,4.0,1.0,7.0,2.0 };

int ja\[17\] = { 0,3,4,1,2,3,5,2,7,3,6,4,5,5,7,6,7 };

int ia\[9\] = {0, 3, 7, 9, 11, 13, 15, 16, 17};

int ierr;

ierr = KmlIssCgInitDI(&handle, n, a, ja, ia);

其他API还有

KmlIssCgSetUserPreconditioner?I
关联用户自定义预条件回调函数，如不使用自定义的预条件，该接口可以不使用。

KmlIssCgSet?I? 设置迭代求解的相关参数。

KmlIssCgSolve?I 求解线性代数方程组。

KmlIssCgGet?I? 获得迭代求解相关参数。

KmlIssCgClean?I 释放内部数据结构。

## 3 GCR求解器

GCR（Generalized Conjugate
Residual，广义共轭残差），一种求解大型非对称稀疏线性方程组的Krylov子空间方法。

KmlIssGcrInit?I 初始化数据结构，并将用户提供的系数矩阵关联到求解系统中。

KmlIssGcrSet?I? 设置迭代求解的相关参数。
