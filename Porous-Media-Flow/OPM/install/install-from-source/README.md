# 源码编译OPM

经过对各版本的编译和测试(使用Norne和Yanchang算例)，目前最佳版本是202110

需要依次编译opm-common, opm-grid, opm-material, opm-models, opm-simulator等,可以不安装opm-upscaling

## 使用amgcl

使用vexcl-1.4.2和amgcl-1.4.3, 使用cmake配置编译时，选择examples和tests编译

设置amg_dir=amgcl的build/cmake路径

### 注意：使用norne算例测试amgcl和使用dune求解器的计算效率, amgcl没有加速求解速度

