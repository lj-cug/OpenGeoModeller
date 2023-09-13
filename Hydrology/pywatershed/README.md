# pywatershed

https://github.com/EC-USGS/pywatershed

GSFLOW已经采用源码编译的方式，耦合了PRMS-mf6，但使用起来非常复杂。

pywatershed项目，使用Python调用PRMS水文模型和modflow6地下水模型。最终，通过BMI耦合器，将PRMS与mf6耦合。

## 特色

Python语言相比C/FORTRAN语言，更容易使用

PRMS与MODFLOW6通过BMI耦合器，提高了代码开发速率

Python与C/FORTRAN语言的混合调用方式，成为近年来复杂模型开发的趋势
