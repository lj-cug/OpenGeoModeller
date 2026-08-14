# ESMF

美国研发的地球耦合模拟框架ESMF

编程语言： FORTRAN

编译版本： 7.1.0, 8.1.1

## 本仓库文档

- [install](./install/)：ESMF / pyESMF 安装与环境变量
- [doc](./doc/)：基于 ESMF 的 ESM 说明
- [esmx-app-prototypes](./esmx-app-prototypes/)：ESMX 示例

## 示例代码
### esmf-training

### esmx-app-prototypes
使用ESMX_Builder apps/basicApp.yaml -g -t方便地构建esmf-app，见 [esmx-app-prototypes](./esmx-app-prototypes/)

## 应用

[RegESM](../RegESM/)

## 参考文献

Arlindo da Silva, et al. The Earth System Modeling Framework.

### ESMF Joint Specification Team

Email: esmf_tech@ucar.edu

## 相关文档

- [ESM-Coupler](../)：学科入口
- [RegESM](../RegESM/)：ESMF 耦合应用
- [Meteorology/WRFV4](../../Meteorology/WRFV4/)：常被耦合的大气分量
- [hpc-base](../../hpc-base/)：编译器、MPI 等
