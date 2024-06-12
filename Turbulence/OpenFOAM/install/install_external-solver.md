# install petsc4foam
petsc4foam就是external-solver-main

## 设置PETSc的环境变量
```
PETSC_ARCH_PATH

```


## 安装
To install in the normal OpenFOAM directories (using `wmake`)
```
./Allwmake
```
will install the library under `FOAM_MODULE_LIBBIN`.

If such variable is not defined, the standard `FOAM_USER_LIBBIN` will be used.

To install into different locations, you can use the `-prefix=PATH` or
`-prefix=shortcut` option (for OpenFOAM-v2006 and newer):
```
./Allwmake -prefix=/install/path

# Installing into the OpenFOAM FOAM_LIBBIN
./Allwmake -prefix=openfoam
```
or specify via the environment.
```
FOAM_MODULE_LIBBIN=/install/path  ./Allwmake
```

## 使用
In order to use the library, two changes are required:
- add the libpetscFoam library to the optional keyword entry libs of the control dict file

      libs       (petscFoam);

- set the name of the solver and preconditioner in each solver of the fvSolution to petsc.
  The options database keys of each PETSc object have to be added in the petsc/options subdict of each solver equation.
  The default behaviour of the library is to convert the matrix from LDU to CSR at each time step.
  However, the user can change the cache update frequency among the following choices:

  - never (none)
  - always
  - periodic
  - adaptive

  The cache update frequency is set for both matrix and preconditioner in the petsc/caching subdict.

An example is reported below. Other examples can be found in the tutorial folder or in the [HPC repo](https://develop.openfoam.com/committees/hpc).
For more details, the user can read the paper [1].

    solvers
    {
        p
        {
            solver          petsc;
            preconditioner  petsc;

            petsc
            {
                options
                {
                    ksp_type    cg;
                    pc_type     bjacobi;
                    sub_pc_type icc;
                }

                caching
                {
                    matrix
                    {
                        update always;
                    }

                    preconditioner
                    {
                        update always;
                    }
                }
            }
        }
    }
