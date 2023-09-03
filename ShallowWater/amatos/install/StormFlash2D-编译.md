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