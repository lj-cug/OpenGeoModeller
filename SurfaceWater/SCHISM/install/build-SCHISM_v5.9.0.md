# 编译SCHISM_v5.9.0及以上版本

## 下载源码

git clone https://github.com/schism-dev/schism.git

cd schism

git checkout tags/v5.10.0  #- for v5.10.0.

## 编译

需要FORTRAN和C编译器(MPI wrapper, 如mpif90, mpicc), NetCDF (version 4.4 and above), Python (version 2.7 and above)

### GNU Make

cp Make.defs.bora Make.defs.myown

ln -sf Make.defs.myown Make.defs.local

编辑Make.defs.local，如MPI编译器名称，NetCDF4的路径(-lnetcdff for FORTRAN)和可执行程序名称(EXEC)

打开/关闭**include_modules**中的相关模块(TVD_LIM必须有值)

确保mk/sfmakedepend.pl和mk/cull_depends.pl有可执行权限的(chmod+x)

cd ../src

make clean

make pschism

注意：如果不是git clone的源码，编译时会出错

最后，生成可执行程序：pschism_* (*代表打开的相关模块名称，如SED etc.)

### CMAKE

需要2个文件：

1. SCHISM.local.build: 打开/关闭选项模块(与include_modules类似). 

TVD_LIM必须有数值, 如果其他模块关闭，则编译水动力模块. 

NO_PARMETIS: 绕过ParMETIS库, 运行时需要提供一个区域分解图partition.prop (与ParMETIS输出的global_to_local.prop一样)

OLDIO: 控制全局输出的开关. 实施异步I/O (aka 'scribed' I/O)，合并全局变量，用scribes核心输出到文件。使用该选项，则要关闭OLDIO，用户需要再命令行指定scribe核心编号，详细信息参考Run-the-model.md；如果打开OLDIO，则使用之前的I/O模式(每个MPI进程都输出)，用户需要使用后处理脚本合并输出.

2. SCHISM.local.cluster_name: 与Make.defs.local类似，该文件定义最重要的环境变量，如编译器和NetCDF库的名称和路径

cp -L SCHISM.local.whirlwind SCHISM.local.myown  # 使用现有的SCHISM.local.cluster_name

mkdir ../build

cd ../build; rm -rf * # Clean old cache

cmake -C ../cmake/SCHISM.local.build -C ../cmake/SCHISM.local.myown ../src/

完成CMAKE配置后，执行：

make -j8 pschism

或者

make VERBOSE=1 pschism # serial build with a lot of messages

最后，生成的可执行程序pschism_*在build/bin/，编译库在build/lib/，工具脚本在bin/
