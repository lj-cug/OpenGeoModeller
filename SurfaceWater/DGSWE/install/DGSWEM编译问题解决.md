# 编译dgswem出现的错误及解决方法

\(1\)

Error: There is no specific subroutine for the generic
\'mpi_dist_graph_creat_adjacent\'

**解决方法**：使用mpich2库(mpich-3.3)，或者使用OpenMPI(5.0版本以上)。

gedit \~/.bashrc

export PATH=/home/lijian/mpich-3.3/bin:\$PATH

\(2\) 对\'dgesv\_\"未定义的应用

没有正确安装或设置lapack库。

打开/work/makefile，阅读第118行下面的内容：

\###############################################################

\# Library Links (Platform Specific) \#

\###############################################################

查看主机名：hostname

修改makefile中的ifeq
(\$(NAME),chl-tilos)为：lijian（我的笔记本的主机名）

\(3\) 使用Python 3执行run_case.py出现：print l.strip()的语法错误

[切换为Python 2.7]{.mark}

alias python=\'/usr/bin/python2.7\'

在bashrc中使用假名，将永久生效。

或者建立软连接：ln --s /usr/bin/python2.7 /usr/bin/python

或者使用update-alternatives

\(4\) 执行./plot出现错误：convert not authorized error/..

修改：/etc/ImageMagick-6/policy.xml文件中的内容：

rights=\"none\" rights=\"read\|write\"

# python的进程管理语法

os.chdir()方法用于改变当前工作目录到指定的路径。

os.rename()方法用于命名文件或目录，从src到dst，如果dst是一个存在的目录，将抛出OSError。

os.rename语法

rename()方法语法格式如下：

os.rename(src, dst)

参数

src \-- 要修改的目录名

dst \-- 修改后的目录名