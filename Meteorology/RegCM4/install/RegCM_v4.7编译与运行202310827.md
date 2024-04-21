# 第1章 版本说明

RegCM_4.7, 新增加：（1）MM5非静水压力动力学核心；（2）修改SAV文件，更好与RegESM框架耦合；（3）地形数据使用GMTED数据集；（4）DUST
12bin；（5）化学和气溶胶模块与CLM4.5兼容；（5）支持从BATS和CLM4.5地表模型的结果初始化土壤湿度。

RegCM_v5.0, 增加了：第3种非静水压力计算核心

## 设置环境变量

export REGCM_ROOT=/home/lijian/RegCM_v4.7

# 第2章 获取代码

http://gforge.ictp.it/gf/project/regcm/frs

./bootstrap.sh需要安装了autoconf, automake and libtool

# 第3章 安装过程

NCO源码：http://nco.sourceforge.net/src

CDO源码：https://code.zmaw.de/projects/cdo/files

A Scientific Plotting and Data Analysis Software such as:

\(1\) IGES GrADS 2.0 Graphical Analysis and Display System. Convenient
helpers are packed in RegCM to use GrADS with RegCM netCDF output files.
Binaries and source code can be obtained from
http://www.iges.org/grads/downloads.html

\(2\) NCL, NCAR CISL Command Language. The NCL can read netCDF output
files, and sample scripts can be found in the Tools/Scripts/NCL
directory. Binaries and source code can be obtained from
http://www.ncl.ucar.edu

## 编译

基本命令：./configure; make; make check; make install

make installcheck

make clean

make distclean

make uninstall

### 编译器与编译参数

[./configure \--help]{.mark}

./configure CC=c99 CFLAGS=-g LIBS=-lposix

./configure CC=icc FC=ifort

./configure CC=\"gcc -arch i386 -arch x86_64 -arch ppc -arch ppc64\" \\

CXX=\"g++ -arch i386 -arch x86_64 -arch ppc -arch ppc64\" \\

CPP=\"gcc -E\" CXXCPP=\"g++ -E\"

## 程序命名

./configure --prefix=PREFIX

\--exec-prefix=PREFIX

make install prefix=/alternate/directory

make install DESTDIR=/alternate/directory

前后缀名：

\--program-prefix=PREFIX

\--program-suffix=SUFFIX

# 第4章 获取全球数据

运行测试算例的第1步就是：获取静态数据，定位模拟区域DOMAIN，以及大气和海洋的全球模型数据集，构建运行局部区域模拟的初始条件和边界条件ICBC。

export ICTP_DATASITE=http://clima-dods.ictp.it/regcm4

RegCM模拟需要的公开数据都能到上面的网址下载。

现在，要将需要用的全局数据下载到当地硬盘上。

尝试使用curl和wget来下载数据。

## 4.1全球数据存储路径

一年的全球数据，容量\>8 Gb

存储在REGCM_GLOBEDAT

\$\> cd \$REGCM_GLOBEDAT

\$\> mkdir SURFACE CLM CLM45 SST EIN15

## 4.2静态的地表数据集

需要下载4个文件：地形、陆地类型分类、湖泊深度（可选）和土壤类型分类（运行DUST需要）

全球数据：30'水平分辨率

![](./media/image1.png)

## 4.3 CLM3.5数据集

如果使用CLM选项，需要全球陆地表面特征数据集的一系列文件：

![](./media/image2.png)

以上文件是clm2rcm程序的输入文件（见6.2节）

## 4.4 CLM4.5数据集

![](./media/image3.png)

以上文件是mksurfdata程序的输入文件（见6.3节）

## 4.5海表温度(SST)

SST数据下载有很多种选择，但为了测试，仅下载CAC
OISST周数据（1981-现在）

![](./media/image4.png)

## 4.6大气和陆地表面温度的全球数据

构建区域尺度模型的初始和边界条件，从一个[全球气候模型（GCM）]{.mark}的输出数据，插值到RegCM网格上。GCM数据可以来自于任何模型，也可下载公开的数据集，现在下载EIN15数据集（比如1990年:Jan 01 00:00:00 UTC to Dec 31 18:00:00 UTC）

![](./media/image5.png)

准备好以上数据，就可以尝试运行RegCM了

# 第5章 运行算例

## 5.1设置运行环境

运行程序的路径：REGCM_RUN

![](./media/image6.png)

接着，修改namelist文件中的内容。设置好连接数据集和程序的路径。

## 5.2使用terrain程序创建DOMAIN文件

第一步：定位模拟区域，创建DOMAIN文件。输入命令：

![](./media/image7.png)

在input路径下，生成如下2个文件：

![](./media/image8.png)

DOMAIN文件包含局部的地形和landuse数据集，还有投影信息和land-sea mask。

第2个文件是ASCII格式的landuse，可以根据需要修改其内容。

查看DOMAIN内容，可以使用GrADSNcPlot程序查看：

./bin/GrADSNcPlot input/test_001_DOMAIN000.nc

## 5.3使用sst程序创建SST

创建模型需要的海表温度文件，读取全球数据集。

![](./media/image9.png)

在input路径下，生成如下1个文件：

![](./media/image10.png)

SST文件包含用于生成namelist文件中指定的时期内的初始和边界条件的海表温度。可以使用GrADSNcPlot程序查看：

./bin/GrADSNcPlot input/test_001_SST.nc

## 5.4使用icbc程序创建ICBC文件

![](./media/image11.png)

在input路径下，生成更多的文件：

![](./media/image12.png)

ICBC文件包含表面压强、表面问对、水平向3D风速分量、3D温度和input文件中指定的时期和时间分辨率上，RegCM计算域内的混合比(mixing
ratio)。

可以使用GrADSNcPlot程序查看：

./bin/GrADSNcPlot input/test_001_ICBC.1990060100.nc

## 5.5第一次运行RegCM模型

![](./media/image13.png)

程序开始运行，屏幕上输出一系列诊断信息。大约10分钟。模拟正常结束，则：RegCM
V4 simulation successfully reached end

output路径下现在有4个文件：

![](./media/image14.png)

ATM文件包含模型输出的大气状态。

SRF文件包含表面诊断变量。

RAD文件包含辐射通量信息。

SAV文件包含模拟周期结束时的状态变量，用于重启。因此，可以将长时段模拟分解成更短的模拟。

查看表面场，输入命令：

./bin/GrADSNcPlot output/test_001_SRF.1990060100.nc

# 第6章 配置自己的模拟

## 6.1 namelist文件

具体可查看README.namelist

coreparam namelist：动力学核心选项

dimparam namelist：计算域维度定义参数

geoparam namelist：地形相关参数

terrainparam namelist：

globdatparam namelist：sst和icbc前处理程序读取

fnestparam namelist

restartparam namelist

timeparam namelist：内部时间步长控制

outparam namelist：模型输出控制

physicsparam namelist：模型物理参数控制，具体参考Reference Manual

dynparam namelist：动力学核心控制参数

referenceatm and nonhydroparam
namelist：控制非静水压力动力学核心的上部辐射边界条件，以及基准状态的垂向剖面曲线形状

hydroparam
namelist：控制消除快波的静水压力核心的分裂显格式，建议不要修改默认参数值

boundaryparam namelist：侧向边界条件

cldparam namelist：控制云所占比例的算法

subexparam namelist：控制SUBEX湿度格式

microparam namelist：控制新的微物理格式

grellparam, emanparam, tiedtkeparam and kfparam namelists：

holtslagparam namelist

uwparam namelist

slabocparam namelist：定义调整混合层模型的海洋q通量的参数和时间间隔

slabocparam namelist：扰动icbc的输入场

tweakparam namelist：调整模型，自定义工况模拟

rrtmparam namelist：RRTM辐射格式

chemparam namelist：化学和气溶胶选项控制

debugparam namelist

## 6.2 CLM3.5选项

## 6.3 CLM4.5选项

## 6.4 敏感性试验

尽管LBC驱动提供了模型的约束条件，RegCM4的特点就是：由于一些非线性过程（如对流）引起的某种程度上的内部变化。

例如，如果对初始条件和侧向边界条件引入小扰动，模型将产生不同的状态变量模式，如降雨模式，当与控制模拟比较时，既是噪声（有时是有一定形式的）。

噪声与计算域大小(domain size)和气象区域(climatic
regime)，例如，特别在暖气象区域（如夏季或热带地区）和较大的计算域的情况。

当修改模型做敏感性试验时，如改变land use，这种内部变化的噪声可能误解为模型对修改因子的响应。

当做敏感性试验时RegCM4用户必须清楚这一点。过滤噪音的最好方法就是：实施ensemble simulations，考察系综平均(ensemble average)结果，从噪声中提取真实的模型响应。

O'Brien, T. A., L. C. Sloan, M. A. Snyder, Can ensembles of regional
climate model simulations improve results from sensitivity studies?,
Climate Dynamics, 37, 1111--1118.

# 第7章 如何启用或不启用一些物理模块

TBD....

# 第8章 后处理工具

RegCM模拟结果保存为NetCDF格式，可以使用很多后处理工具。

## 8.1命令行工具

### 8.1.1 netCDF库工具

ncdump

ncgen

nccopy

### 8.1.2 NetCDF操作工具NCO

![](./media/image15.png)

使用手册见：<http://nco.sourceforge.net/nco.html>

使用范例：

![](./media/image16.png)

![](./media/image17.png)

### 8.1.3气象数据操作工具CDO

## 8.2 GrADS程序

### 8.2.1 GrADS的限制

## 8.3 CISL的NCL：NCAR Command Language

## 8.4 MATLAB工具？

# 参考文献

O'Brien, T. A., L. C. Sloan, M. A. Snyder, Can ensembles of regional
climate model simulations improve results from sensitivity studies?,
Climate Dynamics, 37, 1111-1118.