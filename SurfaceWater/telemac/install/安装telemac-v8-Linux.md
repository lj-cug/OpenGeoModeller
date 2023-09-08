# Install telemac-v8

参考：  http://wiki.opentelemac.org/doku.php?id=installation_on_linux

## 需要的工具

Python 3.7+
Numpy 1.15+
A Fortran compiler (GFortran 4.6.3 is the minimum)
MPI
METIS-5.1
SCOTCH
ParMETIS
PT-SCOTCH
SciPy
Matplotlib
GOTM

## 下载源码

wget https://gitlab.pam-retd.fr/otm/telemac-mascaret/-/archive/v8p3r1/telemac-mascaret-v8p3r1.tar.gz

或者

git clone https://gitlab.pam-retd.fr/otm/telemac-mascaret.git my_opentelemac

cd my_opentelemac

git checkout tags/v8p4r0

## 安装前的说明

<root> 是TELEMAC-MASCARET的源码路径

<systel.cfg> 是构建配置文件

<config> 指向使用的构建配置文件

<pysource> 是编译环境文件

## 设置编译环境

解释如何创建<pysource>文件，你可以找到一个模板文件： <root>/configs/pysource.template.sh

设置如下环境变量：

export HOMETEL=$HOME/telemac-mascaret
export SYSTELCFG=$HOMETEL/configs/systel.cfg
export USETELCFG=gfortranHPC
export SOURCEFILE=$HOMETEL/configs/pysource.gfortranHPC.sh

export METISHOME=~/opt/metis-5.1.0

参考 pysource.sh

可以在~/.bashrc中添加如下：

source $HOME/telemac-mascaret/configs/pysource.gfortranHPC.sh

### 加载环境

source pysource.gfortranHPC.sh

## 配置TELEMAC-MASCARET

解释如何创建<systel.cfg>文件，在<root>/configs下有一些配置文件模板

参考 systel.cfg

## 编译TELEMAC-MASCARET

source pysource.gfortranHPC.sh

显示配置：  config.py

开始编译整个系统： compile_telemac.py

## 运行作业

cd  <root>examples/telemac2d/gouttedo

telemac2d.py t2d_gouttedo.cas --ncsize=4





