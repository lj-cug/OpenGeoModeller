# Customising Firedrake
https://www.firedrakeproject.org/install.html#customising

## Prepared configurations
$ python3 firedrake-configure --show-petsc-configure-options --arch complex

- default: the default installation, suitable for most users
- complex: an installation where PETSc is configured using complex numbers

## Optional dependencies
### SLEPc
To install Firedrake with SLEPc support you should:
1 Pass --download-slepc when running PETSc configure (see Installing PETSc):
```
$ python3 ../firedrake-configure --show-petsc-configure-options | xargs -L1 ./configure --download-slepc
```

Set SLEPC_DIR:
```
$ export SLEPC_DIR=$PETSC_DIR/$PETSC_ARCH
```

Continue with the installation as normal but install Firedrake with the slepc optional dependency. For example:
```
$ pip install --no-binary h5py 'firedrake[check,slepc]'
```

### VTK
To install Firedrake with VTK, it should be installed using the vtk optional dependency. For example:
```
$ pip install --no-binary h5py 'firedrake[check,vtk]'
```

### PyTorch
To install Firedrake with PyTorch, it should be installed using the torch optional dependency. For example:
```
$ pip install --no-binary h5py 'firedrake[check,torch]' --extra-index-url https://download.pytorch.org/whl/cpu
```

Observe that, in addition to specifying torch, an additional argument (--extra-index-url) is needed. More information on installing PyTorch can be found [here](https://pytorch.org/get-started/locally/).

### JAX
To install Firedrake with JAX, it should be installed using the jax optional dependency. For example:
```
$ pip install --no-binary h5py 'firedrake[check,jax]'
```

### Netgen
To install Firedrake with Netgen support, it should be installed with the netgen optional dependency. For example:
```
$ pip install --no-binary h5py 'firedrake[check,netgen]'
```

## Customising PETSc
Since firedrake-configure only outputs a string of options it is straightforward to customise the options that are passed to PETSc configure. You can either:

- Append additional options when configure is invoked. For example, to build PETSc with support for 64-bit indices you should run:
```
python3 ../firedrake-configure --show-petsc-configure-options | xargs -L1 ./configure --with-64-bit-indices
```

- Write the output of firedrake-configure to a file than can be modified. For example:
```
python3 ../firedrake-configure --show-petsc-configure-options > my_configure_options.txt
cat my_configure_options.txt | xargs -L1 ./configure
```

### Reconfiguring an existing PETSc
If rebuilding an existing PETSc installation, rather than removing everything and starting from scratch, it can be useful to modify and run the reconfigure-ARCH.py Python script that PETSc generates. This can be found in $PETSC_DIR/$PETSC_ARCH/lib/petsc/conf. Other example scripts can be found in $PETSC_DIR/config/examples directory.

# Alternative installation methods
Firedrake provides a number of different Docker images that can be found here. The main images best suited for users are:

- firedrake-vanilla-default: a complete Firedrake installation with ARCH default
- firedrake-vanilla-complex: a complete Firedrake installation with ARCH complex
- firedrake: the firedrake-vanilla-default image with extra downstream packages installed

To use one of the containers you should run:
```
$ docker pull firedrakeproject/<image name>:latest
```
to download the most recent image (replacing <image name> with the desired image). Then you can run:
```
$ docker run -it firedrakeproject/<image name>:latest
```
to start and enter a container.

You can also download an image for a specific version by replacing latest with a version tag, for example:
```
$ docker run -it firedrakeproject/<image name>:2025.10.2
```
It is possible to use Microsoft VSCode inside a running container. Instructions for how to do this may be found here.

The Docker daemon runs with superuser privileges and so has the potential to damage your system, in particular if volumes are mounted between the container and host. We therefore strongly advise you to take care when using Docker. More information can be found here.

## Google Colab
Firedrake can also be used inside the browser using Jupyter notebooks and Google Colab. For more information please see here.
