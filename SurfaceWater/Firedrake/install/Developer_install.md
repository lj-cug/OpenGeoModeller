# Developer install
https://www.firedrakeproject.org/install.html#customising

You are currently looking at the documentation for the current stable release of Firedrake. For the most recent developer documentation you should follow the instructions [here](https://www.firedrakeproject.org/install.html#installing-firedrake-using-pip).

In order to install a development version of Firedrake the following steps should be followed. You should decide in advance which development branch that you want to install (main or release, see here for the differences between them).

1 Install system dependencies [as before](https://www.firedrakeproject.org/install.html#install-system-dependencies)

2 Clone PETSc. If you are trying to use the release branch of Firedrake then you should use the PETSc version from firedrake-configure:
```
$ git clone --branch $(python3 firedrake-configure --show-petsc-version) https://gitlab.com/petsc/petsc.git
```
If you are instead building the unstable main branch of Firedrake then the default branch of PETSc (also called main) should be used:
```
$ git clone https://gitlab.com/petsc/petsc.git
```

3 Configure and build PETSc as usual:
```
$ cd petsc
$ python3 ../firedrake-configure --show-petsc-configure-options | xargs -L1 ./configure
$ make PETSC_DIR=/path/to/petsc PETSC_ARCH=arch-firedrake-default all
$ make check
$ cd ..
```

4 Clone the desired branch of Firedrake:
```
$ git clone <firedrake url> --branch <firedrake branch>
```
where <firedrake url> is https://github.com/firedrakeproject/firedrake.git or 
git@github.com:firedrakeproject/firedrake.git 
as preferred and <firedrake branch> is main or release.

5 Set the necessary environment variables:
```
$ export $(python3 firedrake-configure --show-env)
```

6 Create and activate a virtual environment:
```
$ python3 -m venv venv-firedrake
$ . venv-firedrake/bin/activate
```

7 Install petsc4py and Firedrake’s other build dependencies:
```
$ pip cache purge
$ pip install $PETSC_DIR/src/binding/petsc4py
$ pip install -r ./firedrake/requirements-build.txt
```

8 Install Firedrake in editable mode without build isolation:
```
$ pip install --no-build-isolation --no-binary h5py --editable './firedrake[check,docs]'
```

### Editing subpackages
If you want to edit one of Firedrake’s dependencies (e.g. FIAT or UFL) then you should follow an analogous process to the one used to install a developer version of Firedrake above: git clone the repository and then install it in editable mode. However, there are a number of footguns to look out for:

1 The default branch of the subpackage may differ depending on whether you are editing Firedrake main or release. For example, the FIAT main branch is compatible with Firedrake main, and its release branch is compatible with Firedrake release.

To check the branch that you need you should check the pyproject.toml on the relevant Firedrake branch (main, release). On Firedrake main for example you will see:
```
dependencies = [
  # ...
  "firedrake-fiat @ git+https://github.com/firedrakeproject/fiat.git@main",
  # 本地下载的包: firedrake-fiat @ file:///path/to/local/fiat
]
```
which tells you that the main branch of FIAT is expected.

2 These packages must be installed after Firedrake. If Firedrake is installed after installing the subpackage then the subpackage will be overwriiten. This is due to the way that pip manages dependencies. Similarly, it is necessary that dependencies are themselves installed in reverse order. For example, Firedrake depends on both FIAT and UFL, but FIAT also depends on UFL, therefore FIAT must be installed after Firedrake but before UFL.

If you are unsure on the dependency order then, after installing Firedrake, you can use pip to query each package. 
For example, part of the output of pip show firedrake-fiat is:
```
$ pip show firedrake-fiat
...
Requires: fenics-ufl, numpy, recursivenodes, scipy, symengine, sympy
Required-by: firedrake
```

3 If you update your branch of Firedrake it may also be necessary to update the subpackages by running git pull.
