# Install Firedrake
https://www.firedrakeproject.org/install.html#install-system-dependencies

A native installation of Firedrake is accomplished in 3 steps:
```
Install system dependencies
Install PETSc
Install Firedrake
```
## Prerequisites
On Linux the only prerequisite needed to install Firedrake is a suitable version of Python (3.10 or greater).

## firedrake-configure
To simplify the installation process, Firedrake provides a utility script called firedrake-configure. This script can be downloaded by executing:
$ curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/release/scripts/firedrake-configure

Note that firedrake-configure does not install Firedrake for you. It is simply a helper script that emits the configuration options that Firedrake needs for the various steps needed during installation.

This means that if you want to install Firedrake in a non-standard way (for instance with a custom installation of PETSc, HDF5 or MPI) then it is your responsibility to modify the output from firedrake-configure as necessary. This is described in more detail in Customising Firedrake.




# 激活虚拟环境，使用Firedrake
source firedrake/bin/activate

# 并行版本安装
https://www.firedrakeproject.org/parallelism.html

如果你在自己的电脑上，使用并行进程数超过了CPU的真实物理核心数，则要设置以下环境变量：

export PETSC_CONFIGURE_OPTIONS="--download-mpich-device=ch3:sock"

如果你想用自己安装的调优的MPI库, 可以执行： 
python3 firedrake-install --mpiexec=mpiexec --mpicc=mpicc --mpicxx=mpicxx --mpif90=mpif90

# 测试安装
```
cd $VIRTUAL_ENV/src/firedrake
pytest tests/regression/ -k "poisson_strong or stokes_mini or dg_advection"
```

#升级
firedrake-update --help

# 安装终端后恢复更新操作
```
cd src/firedrake
git pull
./scripts/firedrake-install --rebuild-script
You should now be able to run firedrake-update
```

# 构建说明文档
```
python3 firedrake-install --documentation-dependencies

或者

firedrake-update --documentation-dependencies
cd firedrake/firedrake/src/firedrake/docs
make html
```
