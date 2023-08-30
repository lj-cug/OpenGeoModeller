# Obtaining Firedrake

curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install

# Running firedrake-install with no arguments will install Firedrake in a python venv created in a firedrake subdirectory of the current directory

python3 firedrake-install

# a full list of install options

python3 firedrake-install --help

 In particular, you may wish to customise the set of options used to build PETSc. To do so, set the environment variable PETSC_CONFIGURE_OPTIONS before running firedrake-install. You can see the set of options passed to PETSc by providing the flag --show-petsc-configure-options.

# You will need to activate the venv in each shell from which you use Firedrake

source firedrake/bin/activate

# Installing for parallel use https://www.firedrakeproject.org/parallelism.html

export PETSC_CONFIGURE_OPTIONS="--download-mpich-device=ch3:sock"

python3 firedrake-install --mpiexec=mpiexec --mpicc=mpicc --mpicxx=mpicxx --mpif90=mpif90


# Testing the installation

cd $VIRTUAL_ENV/src/firedrake
pytest tests/regression/ -k "poisson_strong or stokes_mini or dg_advection"

# Upgrade

firedrake-update --help

# 安装终端后恢复更新操作

cd src/firedrake

git pull

./scripts/firedrake-install --rebuild-script

You should now be able to run firedrake-update

# Building the documentation

python3 firedrake-install --documentation-dependencies

或者

firedrake-update --documentation-dependencies

cd firedrake/firedrake/src/firedrake/docs

make html

