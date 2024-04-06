# OASIS3-MCT

法国CERFACS开发的耦合模拟器

编程语言：FORTRAN

特点是：模式组件独立编译为可执行程序,然后独立执行：
mpirun -np 6 ./croco : -np 12 ./wrf.exe 

耦合器参数设置在namcouple，其中可定义模式组件之间的交换频率.

## 应用案例

```
大气-海洋-波浪：WRF-CROCO-WW3
大气-陆地-地下水：COSMO-CLM-ParFLOW
```

[安装脚本及说明参考](./Coupling_WRF_CROCO_WW3_with_OASIS)

[WRF-OASIS3-MCT编译指导](./WRF_OASIS3-MCT-build)

## tutorial

[练习示例](./doc/201903_Eric_tutorial.pdf)

## 参考文献

Anthony Craig, Sophie Valcke, and Laure Coquart. 2017. Development and performance of a new version of the OASIS coupler, OASIS3-MCT_3.0. Geosci. Model Dev., 10, 3297–3308.
