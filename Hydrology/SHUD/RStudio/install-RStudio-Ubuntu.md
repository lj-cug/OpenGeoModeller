# install-RStudio-in-Ubuntu 20.04

## 安装R-base

apt update -qq

apt install --no-install-recommends software-properties-common dirmngr
 
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
 
add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"

安装R：  apt install --no-install-recommends r-base

# 安装RStudio

R Studio网站已经正式改名为Posit。网址：https://posit.co/

但R Studio软件仍为R Studio。点击右上角 DOWNLOAD RSTUDIO

页面跳转，下拉界面选择Free版本，点击DOWNLOAD

页面跳转，找到Ubuntu 22对应的版本（rstudio-2022.07.2-576-amd64.deb），点击下载。

下载好后进入deb文件所在文件夹，在当前文件夹打开终端，运行安装命令：

sudo dpkg -i   rstudio-2022.07.2-576-amd64.deb

遇到错误，运行代码纠正：

apt --fix-broken install

或者

apt-get install -f

再次安装：

sudo dpkg -i   rstudio-2022.07.2-576-amd64.deb

安装成功。

## 更改为国内源

检查当前的源：   getOption("repos")

第一步，选择Tools选项中的Global Options选项。

第二步，选择Packages选项中的change按钮，进行镜像站点的选择。

也可以输入以下代码（适合不带图形界面的R），直接修改：

options(repos=structure(c(CRAN="https://mirrors.tuna.tsinghua.edu.cn/CRAN/")))

通过 getOption("repos") 命令可以知道目前的镜像网站是哪里的

## 修改 bioconductor 的安装源

绝大部分的生物信息相关的R包（如DESeq2, limma, clusterProfiler）都在 bioconductor，并不在官方的源里面，所以通过 install.packages() 命令会找不到对应的R包。得使用如下命令安装：
```
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("DESeq2")
```

同样，使用option命令修改bioconductor的源为国内源，就再也不用忍受bioconductor 的龟速了，代码如下：

options(BioC_mirror="https://mirrors.tuna.tsinghua.edu.cn/bioconductor")