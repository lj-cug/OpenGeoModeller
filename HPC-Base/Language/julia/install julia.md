# Windows 系统下安装

从 https://julialang.org/downloads/ 下载 Windows Julia 安装程序。

# Linux/FreeBSD 安装

wget https://mirrors.tuna.tsinghua.edu.cn/julia-releases/bin/linux/x86/1.7/julia-1.7.2-linux-i686.tar.gz --no-check-certificate

解压：

tar zxvf julia-1.7.2-linux-i686.tar.gz

解压完成后将 julia 的解压目录移动到 /usr/local 目录下：

mv julia-1.7.2 /usr/local/

移动完成后我们就可以使用 julia 的完整目录执行 Julia 命令：

# /usr/local/julia-1.7.2/bin/julia -v   
julia version 1.7.2

julia -v 命令用于查看版本号。

julia 使用完整路径调用可执行文件：/usr/local/julia-1.7.2/bin/julia -v

也可以将 julia 命令添加到您的系统 PATH 环境变量中，编辑 ~/.bashrc（或 ~/.bash_profile）文件，在最后一行添加以下代码：

export PATH="$PATH:/usr/local/julia-1.7.2/bin/"
添加后执行以下命令，让环境变量立即生效：

source ~/.bashrc 
    
或

source ~/.bash_profile这样我们就可以直接执行 julia 命令而不需要添加完整路径：

julia -v

julia version 1.7.2

