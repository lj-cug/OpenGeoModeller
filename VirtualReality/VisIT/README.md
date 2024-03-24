# visit

[介绍主页](https://visit-dav.github.io/visit-website/)

## 安装

https://visit-sphinx-github-user-manual.readthedocs.io/en/v3.2.2/gui_manual/Intro/Installing_VisIt.html

### 脚本安装

https://sd.llnl.gov/simulation/computer-codes/visit/executables

``
chmod +x visit-install
visit-install 3.1.4 linux-x86_64 /usr/local/visit
``

### 源码安装
```
./build_visit3_0_1
./build_visit3_0_1 --makeflags -j4
./build_visit3_0_1 --hdf5 --silo
./build_visit3_0_1 --thirdparty-path /usr/gapps/visit/third_party
./build_visit3_0_1 --optional --tarball visit3.0.1.tar.gz
```

## 运行

/usr/local/visit/bin/visit

## FVCOM/SCHISM使用visit后处理

SCHISM在visit中有[插件](https://github.com/schism-dev/schism_visit_plugin)

git clone https://github.com/l3-hpc/visit-scripts.git
