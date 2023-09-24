# pyGIMLi

pyGIMLi是一款高效的使用Python语言编写的地球物理反演工作

地震层析反演，可作为FWI的初始速度模型

[github源码仓库链接](https://github.com/gimli-org/gimli)

[安装](https://www.pygimli.org/installation.html#sec-install)

[Tutorials-How to use pyGIMLi](https://www.pygimli.org/_tutorials_auto/index.html)

[Examples](https://www.pygimli.org/_examples_auto/index.html)
其中，有地震层析反演的例子

## 安装

conda create -n pg -c gimli -c conda-forge pygimli=1.4.3

conda activate pg

或者，
到 https://anaconda.org/gimli/pygimli/files直接下载某版本的tar.gz2文件，然后执行：
 
   conda install package.tar.gz

或者，安装最新版本

git clone https://github.com/gimli-org/gimli

cd gimli

make pygimli J=2

设置环境变量：

export PYTHONPATH=$PYTHONPATH:$HOME/src/gimli/gimli/python

export PATH=$PATH:$HOME/src/gimli/build/lib

export PATH=$PATH:$HOME/src/gimli/build/bin

## 测试

python -c "import pygimli; pygimli.test(show=False, onlydoctests=True)"