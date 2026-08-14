# PIHM

基于求解常微分方程的sundials库，使用三角形非结构网格，实施分布式水文模拟降雨-径流等过程。

编程语言： C

OpenMP并行

数据同化

## 本仓库文档

- [install](./install/)：MM-PIHM 等安装说明
- [doc](./doc/)：原理、输入文件与操作说明
- [PIHMgis-Linux](./PIHMgis-Linux/)：Ubuntu 下编译的 PIHMgis 3.0

## 输入数据处理

PIHM输入文件格式，可参考： https://blog.csdn.net/qq_44246618/article/details/115049620

分布式水文模型都需要大量不同类型的输入数据，处理过程繁琐

PIHM模型的快速的数据处理，可联系舒乐乐课题组(PIHM的开发课题组人员):

https://www.shud.xyz/zh/

## PIHMgis-Linux

PIHMgis 3.0源代码中仅有Windows和MacOS的可执行程序

在Ubuntu 20.04系统下编译了PIHMgis 3.0

使用步骤可参考： https://blog.csdn.net/qq_44246618/article/details/115011089

## 参考文献

PIHM Model

## 相关文档

- [Hydrology](../)：学科入口
- [SHUD](../SHUD/)：同源数据处理与建模材料
- [PIHMgis-Linux](./PIHMgis-Linux/)：Linux 下 PIHMgis
- [hpc-base](../../hpc-base/)：编译与运行环境
