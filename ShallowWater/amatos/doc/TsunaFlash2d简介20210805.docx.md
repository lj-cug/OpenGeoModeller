# TsunaFlash2d模型的原理及编译

## 程序结构

主程序Flash90.F90，调用子程序：

[fvm_initialize(p_grid, p_contr)]{.mark}

[fvm_timestepping(p_grid, p_contr)]{.mark}

[fvm_finish(p_grid, p_contr)]{.mark}

[grid_terminate]{.mark}

## 基本原理

Hanert E. et al., 2005

## 编译

运行命令：./FLASH -b -f Parameters.dat

编译OpenMP版本的AMATOS2d_parallel_omp，运行程序后出现错误：

A pointer passed to DEALLOCATE points to an object that cannot be
deallocated

## 参考文献

Hanert E., Le Roux, D. Y., Legat V. Deleersnijder, 2005 An efficient
Eulerian finite element method for the shallow water equations. Ocean
Model. 10, 115-136.
