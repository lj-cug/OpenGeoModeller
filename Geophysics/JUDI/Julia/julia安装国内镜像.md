# julia安装国内镜像

## 使用方式

只需要设置环境变量 JULIA_PKG_SERVER 即可切换镜像。若成功切换镜像，则能通过 versioninfo() 查询到相关信息，例如：

julia> versioninfo()

若不设置该环境变量则默认使用官方服务器 pkg.julialang.org 作为上游。

## 临时使用

不同系统和命令行下设置环境变量的方式各不相同，在命令行下可以通过以下方式来临时修改环境变量

### Linux Bash

export JULIA_PKG_SERVER=https://mirrors.cernet.edu.cn/julia

### Windows Powershell

$env:JULIA_PKG_SERVER = 'https://mirrors.cernet.edu.cn/julia'

也可以利用 JuliaCN 社区维护的中文本地化工具包 JuliaZH 来进行切换：

using JuliaZH              # 在 using 时会自动切换到国内的镜像站

JuliaZH.set_mirror("BFSU") # 也可以选择手动切换到 BFSU 镜像

JuliaZH.mirrors             # 查询记录的上游信息

## 永久使用

不同系统和命令行下永久设定环境变量的方式也不相同，例如 Linux Bash 下可以通过修改 ~/.bashrc 文件实现该目的：

# ~/.bashrc

export JULIA_PKG_SERVER=https://mirrors.cernet.edu.cn/julia
