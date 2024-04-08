# install ESMF

https://earthsystemmodeling.org/docs/release/ESMF_5_2_0p1/ESMF_usrdoc/node9.html

The ESMF build system offers the standard install target to install all necessary files created during the build process into user specified locations. The installation procedure will also install the ESMF documentation if it has been built successfully following the procedure outlined above.

The installation location can be customized using six ESMF_ environment variables:

```
ESMF_INSTALL_PREFIX - prefix for the other five variables.
ESMF_INSTALL_HEADERDIR - where to install header files.
ESMF_INSTALL_LIBDIR - where to install library files.
ESMF_INSTALL_MODDIR - where to install Fortran module files.
ESMF_INSTALL_BINDIR - where to install application files.
ESMF_INSTALL_DOCDIR - where to install documentation files.
```

install ESMF with the command:

make install

Check the ESMF installation with the command:

make installcheck

Advice to installers. Complete the installation of ESMF by defining a single ESMF specific environment variable, named ESMFMKFILE. This variable shall point to the esmf.mk file that was generated during the installation process. Systems that support multiple ESMF installations via management software (e.g. modules, softenv, ...) shall set/reset variable ESMFMKFILE as part of the configuration.

By default file esmf.mk is located next to the ESMF library file in directory ESMF_INSTALL_LIBDIR. Consequently, unless esmf.mk has been moved to a different location after the installation, the correct setting for ESMFMKFILE is $(ESMF_INSTALL_LIBDIR)/esmf.mk.

## 设置环境变量

```
ESMF_BOPT  - g (debug)  - O2 (optimized)
ESMF_COMM  - mpich2, openmpi, intelmpi
ESMF_COMPILER
ESMF_DIR

ESMF_LAPACK              Possible value: not set, "OFF", "mkl", "netlib", "scsl".
ESMF_LAPACK_LIBPATH      Typical value: /usr/local/lib (no default).
ESMF_LAPACK_LIBS         Typical value: "-llapack -lblas" (default is system dependent).

ESMF_NETCDF="standard"   Possible value: not set (default), "split", "standard".
ESMF_NETCDF_INCLUDE
ESMF_NETCDF_LIBPATH
ESMF_NETCDF_LIBS = "-lnetcdff -lnetcdf_c++ -lnetcdf"

ESMF_PNETCDF="standard"  Possible value: not set (default), "standard".
ESMF_PNETCDF_INCLUDE     Typical value: /usr/local/include (no default).
ESMF_PNETCDF_LIBPATH     Typical value: /usr/local/lib (no default).
ESMF_PNETCDF_LIBS="-lpnetcdf"

ESMF_PIO = "internal"    Possible value: not set (default), "internal".

ESMF_XERCES="standard"   Possible value: not set (default), "standard", <userstring>
ESMF_XERCES_INCLUDE      Typical value: /usr/local/include (no default).
ESMF_XERCES_LIBPATH      Typical value: /usr/local/lib (no default).
ESMF_XERCES_LIBS         Typical value: "-lxerces-c" (no default).

ESMF_CXX
ESMF_CXXCOMPILEOPTS
ESMF_CXXCOMPILER
ESMF_CXXLINKDIRS
ESMF_CXXLINKLIBS
ESMF_CXXLINKOPTS
ESMF_CXXLINKER
ESMF_CXXOPTFLAG
ESMF_DEFER_LIB_BUILD     Possible value: ON (default)并行编译, OFF

ESMF_F90
ESMF_F90COMPILEOPTS
ESMF_F90COMPILER
ESMF_F90IMOD
ESMF_F90LINKDIRS
ESMF_F90LINKLIBS
ESMF_F90LINKER
ESMF_F90OPTFLAG

```
