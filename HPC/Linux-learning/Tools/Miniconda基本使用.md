# MiniConda安装程序下载

Anaconda 的安装程序很大，推荐大家使用MiniConda

[Anaconda 镜像下载](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)

[MiniConda 镜像下载](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/)


# 加速MiniConda环境中程序下载速度

Anaconda Prompt使用以下方法将清华镜像添加到anaconda，执行如下命令：

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ 

conda config --set show_channel_urls yes

# conda基本命令

## 设置conda的环境变量（开机启动conda）

1. gedit ~/.bashrc
2. add "export PATH=$PATH:/home/root/anaconda3/bin"
3. save
4. source ~/.bashrc

## 查看当前conda所有环境
conda info --envs

## 创建环境

conda create --name envname python=version

例如：

conda create --name project_verson3.8 python=3.8

注：如果不指定python版本默认安装最新版


## 导出已有的环境 

conda env export > environment.yaml

## 根据yaml文件创建新的环境 

conda env create -f environment.yaml

## 导出当前环境下所使用的包，生成requirements.txt

conda list -e > requirements.txt  #导出当前环境所有的依赖包及其对应的版本号

## 安装requirements.txt中的包

conda install --yes --file requirements.txt   #在新的环境中安装导出的包


## 激活你的环境

conda activate 环境名

## 在你的环境中用conda或者pip安装包

conda install 包名称

或者pip install 包名称 -i https://pypi.tuna.tsinghua.edu.cn/simple（清华镜像）

例如：pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

可以通过conda install，pip install 来安装python程序包。
执行：pip install -r requirements.txt      安装依赖包。

## 查看环境中现有的包：

conda list

pip list


## 退出当前环境

conda deactivate 环境名


## 删除环境

conda remove -n 环境名 --all


## 修改虚拟环境名（在已有的虚拟环境的基础上制造一个新的虚拟环境）

1 新建一个环境，克隆源环境
conda create --name newName --clone oldName

2 删除源环境
conda remove -n oldName --all

3查看现有全部虚拟环境
conda env list
