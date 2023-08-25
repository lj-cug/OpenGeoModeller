# MODFLOW6-6.4.2并行版本源码编译记录

## Prerequisites

Gfortran/Intel FORTRAN

miniconda

Python 3.8+

### 安装编译环境

conda env create -f environment.yml

conda env update -f environment.yml

### 依赖工具

\- \`meson\`

\- \`fprettify\`

\- \`pymake\`

\- \`flopy\`

## 编译串行modflow6源码

meson setup builddir \--prefix=\$(pwd) \--libdir=bin \#
生成ninja配置文件

meson compile -C builddir \# 编译源码

meson install -C builddir \# 生成modflow6可执行程序

生成的可执行程序在bin路径下。执行meson install后也就更新了可执行程序。

## 测试程序

modflow6测试是基于pytest

### 构建开发模式的可执行程序

执行：python autotest/build_exes.py

或者，

在autotest路径下执行：pytest build_exes.py

可设置生成可执行程序的路径，使用参数 \--path

### 重新构建和安装发布的可执行程序

（1）已发布了很多modflow6的可执行程序：

https://github.com/MODFLOW-USGS/executables

（2）执行：python autotest/get_exes.py

或者，

在autotest路径下执行：pytest get_exes.py

（3）生成的可执行程序在bin/rebuilt或bin/downloaded

可设置生成可执行程序的路径，使用参数 \--path

### 更新flopy插件

(1)执行：autotest/update_flopy.py

注意：如果已经从源码安装了当地版本的flopy，执行此不会覆盖已有安装。

(2)另一种方法，使用定义文件DFN更新，默认的DFN文件在doc/mf6io/mf6ivar/dfn，执行：

python autotest/update_flopy.py

python autotest/update_flopy.py doc/mf6io/mf6ivar/dfn

### 外部模型仓库

\[\`MODFLOW-USGS/modflow6-testmodels\`\](<https://github.com/MODFLOW-USGS/modflow6-testmodels>)

\[\`MODFLOW-USGS/modflow6-largetestmodels\`\](<https://github.com/MODFLOW-USGS/modflow6-largetestmodels>)

\[\`MODFLOW-USGS/modflow6-examples\`\](<https://github.com/MODFLOW-USGS/modflow6-examples>)

这些仓库应该位于与modflow6相同的父目录。如果位于其他位置，需要设置环境变量REPOS_PATH，指向父目录。

### 测试模型

git clone MODFLOW-USGS/modflow6-testmodels

git clone MODFLOW-USGS/modflow6-largetestmodels

### 示例模型

git clone MODFLOW-USGS/modflow6-examples

示例模型在克隆后需要一些设置，执行：

cd modflow6-examples/etc

pip install -r requirements.pip.txt

然后，在etc文件夹，运行：

python ci_build_files.py

这样就会构建测试需要使用的例子。

### 运行测试

在autotest文件夹运行pytest。如果需要运行某个文件，显示参数输出，使用：

pytest -v \<file\>

pytest -v -n auto

-n表示并行进程数目；auto表示pytest-xdist在每个处理器上使用1个worker。

### 选择带标识的测试 {#选择带标识的测试 .标题3}

pytest.ini提供标识，包括：slow, repo, large

使用-m \<marker\>选项使用标识，与逻辑and, or, not联合使用。例如：

pytest -v -n auto -m \"not slow and not regression\"

pytest -v -n auto -S

### 外部模型测试

pytest -v -n auto -m \"repo\"

pytest -v -n auto -m \"large\"

\# MODFLOW 6 test models

pytest -v -n auto test_z01_testmodels_mf6.py

\# MODFLOW 5 to 6 conversion test models

pytest -v -n auto test_z02_testmodels_mf5to6.py

\# models from modflow6-examples repo

pytest -v -n auto test_z03_examples.py

\# models from modflow6-largetestmodels repo

pytest -v -n auto test_z03_largetestmodels.py

### 编写测试模型

参考DEVELOPER.md

## 编译并行化modflow6需要的MPI与PETsc

并行化modflow6依赖第3方库：MPI与PETSc

使用源码编译MPI和PETSc:

\$ ./configure \--download-openmpi \--download-fblaslapack

\$ make all

### 使用pkg-config检查PETSc安装

pkg-config \--libs petsc

检查下列路径内容：

\$PETSC_DIR/\$PETSC_ARCH/lib/pkgconfig/

确认存在\*.pc文件。

MPI也有类似的pkgconfig文件夹。

为连接所有的库，这些文件夹路径都要添加到PKG_CONFIG_PATH变量，这样pkg-config可以找到安装的库。

## 编译并行版本的MODFLOW6

## 测试并行版的MODFLOW6

meson setup builddir -Ddebug=false -Dparallel=true \--prefix=\$(pwd)
\--libdir=bin

meson install -C builddir

meson test \--verbose \--no-rebuild -C builddir

(1) 重新配置，在meson setup步骤中添加\--reconfigure

(2) 设置好PKG_CONFIG_PATH环境变量后，自动链接到PETSc和MPI。

使用ldd工具检查可执行程序mf6pro是否成功链接到外部依赖库。

## 测试并行的modflow6

在autotest路径下：

\$ pytest -s \--parallel test_par_gwf01.py

## 调试

确保使用-Ddebug=true编译源码

并行模式下，程序使用PETSc求解器和一个配置文件.petscrc，一并存在于和mfsim.nam相同文件夹下。在.petscrc文件中，增加如下选项：

-wait_dbg

然后启动并行程序，例如使用2核：

mpiexec -np 2 mf6 -p

## 兼容性

| Operating System      |   Toolchain |      MPI      |  PETSc | Package Manager |
|-----------------------|-------------|---------------|--------|-----------------|
| WSL2 (Ubuntu 20.04.5) | gcc 9.4.0   | OpenMPI 4.0.3 | 3.18.2 | NA              |
| Ubuntu 22.04          | gcc 9.5.0   | OpenMPI 4.1.4 | 3.18.5 | NA              |
| Ubuntu 23.04          | gcc 13      | OpenMPI 4.1.4 | 3.18.1 | apt             |
| macOS 12.6.3          | gcc 9.5.0   | OpenMPI 4.1.4 | 3.18.5 | NA              |
| macOS 12.6.6          | gcc 13.1.0  | OpenMPI 4.1.5 | 3.19.1 | Homebrew        |
