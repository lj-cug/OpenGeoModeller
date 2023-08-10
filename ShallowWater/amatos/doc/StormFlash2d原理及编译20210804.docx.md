# StormFlash2D模型介绍

## StormFlash2D子程序介绍

主程序：TAM_main.F90，主程序调用一系列子程序：

io\_\*()

[fvm_initialize]{.mark} ! 初始化网格，设置初始条件，等

[fvm_timestepping]{.mark} ! 核心程序，时间步推进

[fvm_finish]{.mark} ! 结束计算，释放内存

[grid_terminate]{.mark} ! 结束网格生成器

-   ADV_dg.F90，包含DG模型的主程序。

-   DG_CFL\_\*.F90,根据线性理论或局部波速，计算CFL条件数。

-   DG_equation\_\*.F90，定义方程中的所有变量，类似netCDF的CF格式。

-   DG_error\_\*.F90，误差评估，有很多种方式。

-   DG_flux\_\*.F90，计算数值通量，其中有可能需要计算边界条件，DG_boundary\_\*.F90
    > ([SUBROUTINE compute_bc]{.mark})

-   DG_initial\_\*.F90，设置初始条件，很多中算例的初始条件不同。

-   DG_limiter\_\*.F90，多种不同类型的DG坡度限制器。

-   DG_RS\_\*.F90，几种黎曼求解算法，如exact, HLL, HLLC, Roe, Rusanov

-   DG_storm_drag\_\*.F90，几种风暴潮情况下的水面风切应力计算公式

-   DG_storm_holland.F90, Holland台风经验模型

-   DG_time\_\*.F90,
    > 2种时间推进格式（隐格式和显格式）,隐格式的Rosenbrock-Wanner法和显格式的龙格-库塔法（SSP）

-   IO\_\*.F90，管理输入输出的函数

-   MISC_bathy\_\*.F90，地形初始化子程序，可读取netCDF和ETOPO1两种格式

-   MISC_diag\_\*.F90，诊断函数

-   MISC_eqsource.F90，方程源项加载，如地震海啸模拟中的断裂抬升。

-   MISC_linSysSolv.F90，隐格式时间推进中的GMRES求解器。

-   MISC_quad.F90, 定义积分法则的类型

## 编译

StormFlash2d需要的第三方库有：netCDF, LAPACK95, BLAS, LAPACK；

Stormflash2D 的安装过程与安装amatos类似。编译步骤如下：

（1）cd进入路径trunk/compile/\[architecture\]；手动修改或设置Makefile，主要是指定访问第三方库的路径，如NetCDF,
LAPACK, BLAS.

（2）make maincopy (maincopy 是算例的名字)，成功编译后将生成 DGM
可执行程序。

（3）设置环境变量LD_LIBRARY_PATH，指向有libamatos.so的路径，这样可正确连接动态链接库。

（4）执行命令：make datacopy，这样就拷贝了运行测试算例需要的数据文件。

（5）执行命令：./DGM -f Parameters.dat，这样就运行算例程序了。

另外，可以使用Paraview可视化模拟结果文件，如Flash90_nnnnnn.vtu

## 算例

Storms算例：

Test Cases from \"Finite Volume Methods for the Multilayer Shallow Water
Equations with Applications to Storm Surges\", Mandli 2011

112页

SWE算例：

Test cases from \"Shock-Capturing Methods for Free-Surface Shallow
Flows\", Toro 2001 (page 120)
