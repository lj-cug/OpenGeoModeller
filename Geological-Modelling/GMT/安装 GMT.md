# Ubuntu/Debian 下安装 GMT

由于 Ubuntu 和 Debian 自身的软件更新规则，官方软件源提供的通常都是老版本 GMT，比如 GMT 5.4.5 或者 GMT 6.0.0，而不是 GMT 的最新版本。

针对这种情况，有如下几种选择：

使用老版本 GMT

通过 conda 安装 GMT 【推荐】

Linux/macOS 下编译 GMT 源码 【推荐】

通过如下命令，可以安装 Ubuntu/Debian 官方源提供的 GMT 二进制包。

更新软件包列表:
$ sudo apt update

安装 GMT:
$ sudo apt install gmt gmt-dcw gmt-gshhg

安装 GMT 相关工具

生成 PDF、JPG 等图片格式需要 Ghostscript（必须）:
$ sudo apt install ghostscript

地理空间数据格式转换工具 GDAL（必须，未安装则无法使用高精度地形数据）:
$ sudo apt install gdal-bin

制作 GIF 格式的动画需要 GraphicsMagick（可选）:
$ sudo apt install graphicsmagick

制作 MP4、WebM 格式的动画需要 FFmpeg（可选）:
$ sudo apt install ffmpeg


# 通过 conda 安装 GMT

conda 是由 Anaconda 提供的一个跨平台软件包管理器。conda 的 conda-forge 频道提供了 GMT 安装包。使用 conda 安装 GMT 的优点有：跨平台、安装简单、版本切换方便等。
Anaconda 用户可以直接通过以下命令安装、升级以及卸载 GMT。未安装 Anaconda 的用户可以参考《Anaconda 简易教程》安装 Anaconda。
Anaconda 的 base 环境下默认安装了数百个软件包，其中部分软件包与 GMT 存在冲突，导致 base 环境下 GMT 会出现运行报错的问题。Anaconda 用户必须新建并激活一个新环境安装使用 GMT。
我们推荐用户使用只提供了必须依赖包的 miniconda，以节省硬盘空间并且避免 base 环境下的 GMT 运行问题。


## 安装 GMT

安装最新的 GMT 稳定版本:

$ conda install gmt -c conda-forge

也可以安装 GMT 开发版本（该开发版本会每隔几周更新一次）:
$ conda install gmt -c conda-forge/label/dev

## 测试安装

$ gmt --version

6.4.0

## 升级 GMT

GMT 新版本发布后，执行如下命令升级 GMT:

$ conda update gmt

## 卸载 GMT

执行如下命令可以卸载 GMT:

$ conda remove gmt
