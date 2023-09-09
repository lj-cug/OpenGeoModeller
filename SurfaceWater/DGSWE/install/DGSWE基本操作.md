# DGSWE基本操作

见run_case.py代码：

使用python2语言的subprocess module控制各程序的运行：

show_output显示执行某命令（进程）的输出信息。

def show_output(command):

output = subprocess.Popen(command,stdout=subprocess.PIPE,shell=True)

while output.poll() is None:

l = output.stdout.readline()

print l.rstrip(\'\\n\')

## 1、设置绝对路径

plot_work_dir

dgswe_work_dir

example_dir

nprocessors #启用的进程数(MPI)

## 2、编译DG-SWEM

（1）进入/work目录，编译dgswe_mpi程序：

make clean

make metis

make dgswe OPTS=mpi

make dgprep

make dgpost

注意：上述编译遇到错误，解决方法见"DGSWEM编译问题解决.md"

（2）进入/plot/work目录，编译plot程序：

make clean

make plot

（3）运行dgswe

-   建立一个文件：np.in，内容是nprocessors

-   执行前处理：./dgprep \< np.in

-   执行主程序：mpirun --n nprocessors ./dgswe_mpi

-   再执行后处理：./dgpost

（4）绘制（可视化）结果：

./plot

## 3、删除所有运行例子产生的文件

clean_case.py

删除所有的文件，包括可执行程序。

## plot后处理程序

输出为PostScript的矢量图，包括高阶FEM的可视化方法（见Brus,
2017的博士论文）

需要输入文件：plot.inp和plot_sta.inp (可选)

plot.inp的几个感兴趣的参数：

station plot option 绘制测站位置的示意图

order of nodal set for plotting straight elements

order of nodal set for plotting curved elements

adaptive plotting option

colormap path 可选择Legend颜色模式

plot Google Map 下载谷歌地图图片，作为背景图

## error分析程序

分析收敛性和误差的后处理程序，比较粗网格和细网格的收敛速率。

需要输入文件error.inp

!/home/sbrus/Codes/dgswe/grids/converge_quad.grd ! coarse grid file

!/home/sbrus/data-drive/converge_quad/mesh1/P2/CTP2/ ! coarse output
directory

!2 ! p - coarse polynomial order

!2 ! ctp - coarse parametric coordinate transformation order

!.5d0 ! dt - coarse timestep

!/home/sbrus/Codes/dgswe/grids/converge_quad2.grd ! fine grid file

!/home/sbrus/data-drive/converge_quad/mesh2/P2/CTP2/ ! fine output
directory

!2 ! p - fine polynomial order

!2 ! ctp - fine parametric coordinate transformation order

!.25d0 ! dt - fine timestep

!2d0 ! tf - final time (days)

!20 ! lines - lines in output files

## bathy_interp程序

地形的高阶插值计算。

输入文件：bathy.inp；内容？

输出文件：\_interp.hb、elem_nodes.d、interp_nodes.d、boundary_nodes.d、bathy.d

## rimls程序

输入文件：rimls.inp

具体功能尚未清楚。是优化网格的？

/home/sbrus/Codes/dgswe/grids/dummy.cb ! curved boundary file

/home/sbrus/Codes/dgswe/rimls/work/Rimls_test-sub.grd ! eval grid - used
to determine rimls surface evaluation points

1 ! eval hbp - bathymetry order

1 ! eval ctp - parametric coordinate transformation order

/home/sbrus/Codes/dgswe/grids/dummy.cb ! curved boundary file

3 ! lsp - moving least squares fit order

0 ! basis_opt - basis for least squares polynomial (1 - orthonormal,
else - simple)

1d0 ! Erad - radius of Earth

0d0,0d0 ! lambda0,phi0 - center of CPP coordinate system

3.0d0 ! r - muliplier for search radius (1.5 - 4.0)

1.5d0 ! sigma_n - smoothing parameter (0.5 - 1.5)

../output/ ! output directory

1 ! nrpt - number of random points (for converging channel hardwire)

0d0

## spline程序

输入文件：spline.inp

## stations程序

输入文件：dgswe.inp

绘制测站的位置示意图。

## util文件夹

该文件夹下包含很多工具小程序，FORTRAN和MATLAB语言。