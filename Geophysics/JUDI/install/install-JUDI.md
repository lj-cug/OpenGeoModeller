# Install JUDI

## install Julia

wget -c https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.2-linux-x86_64.tar.gz

tar zxvf julia-1.9.2-linux-x86_64.tar.gz

```
gedit ~/.bashrc
export PATH=$PATH:/home/lijian/HPC_Build/Devito/julia-1.9.2/bin
source ~/.bashrc
```

## run julia

julia


## 标准安装

方式1：

pkg> add JUDI

方式2：

julia -e 'using Pkg;Pkg.add("JUDI")'

## 开发者安装

基于自己的[devito](https://www.devitoproject.org/devito/download.html),允许更多的控制.
为考虑与Julia的兼容性,建议使用pip安装devito

指定PYTHON环境变量,这样Julia知道在哪找到python软件包,如devito及其依赖库,执行命令:

export PYTHON=$(which python) # '$(which python3)'

pkg> build PyCall   # rebuild PyCall to point to your python

julia -e 'using Pkg;Pkg.build("PyCall")

## 安装JUDI的其他依赖库

julia PATH/TO/JUDI/deps/install_global.jl

## 测试安装

安装了JUDI，但每个逐个安装依赖，需要使用julia运行所有的examples/tests/experiments

在JUDI路径下：julia --project

在其他路径下：julia --project=path_to_JUDI

如果已经安装了依赖库，
可使用基本的julia命令运行所有脚本： julia script.jl

### 测试

You can run the standard JUDI (julia objects only, no Devito):

GROUP="JUDI" julia test/runtests.jl  # or GROUP="JUDI" julia --project test/runtests.jl

or the isotropic acoustic operators' tests (with Devito):

GROUP="ISO_OP" julia test/runtests.jl  # or GROUP="ISO_OP" julia --project test/runtests.jl
