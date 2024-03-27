# install pyroms

   * Python >= 3.4 (Python 3.8 currently recommended for new environments)
   * [numpy](https://numpy.org/)
   * [scipy](https://www.scipy.org/)
   * [matplotlib](https://matplotlib.org/)
   * [basemap](https://matplotlib.org/basemap/)
   * [netcdf4](https://unidata.github.io/netcdf4-python/netCDF4/index.html)
   * [cftime](https://unidata.github.io/cftime/)
   * [lpsolve55](https://github.com/chandu-atina/lp_solve_python_3x)

   The following is optional: Pyroms can be built and run without it but some of the functionality will be missing.

   * scrip, a Python implementation of [SCRIP](https://github.com/SCRIP-Project/SCRIP),
     the Spherical Coordinate Remapping and Interpolation Package. This is used by the pyroms
     module. The Python scrip code (a rather old version) is
     [bundled in pyroms](https://github.com/ESMG/pyroms/tree/python3/pyroms/external/scrip)
     and can be built and installed separately as described below. In future we plan to
     move from the bundled scrip code to a stand-alone package like
     [ESMF/ESMPy](https://www.earthsystemcog.org/projects/esmpy/) or
     [PySCRIP](https://github.com/dchandan/PySCRIP).
	 
	 The following is optional and provides high-resolution coastlines for basemap:

   * [basemap-data-hires](https://anaconda.org/conda-forge/basemap-data-hires/)
   
## Install from source

```
$ git clone https://github.com/ESMG/pyroms.git
$ pip install -e pyroms/pyroms
$ pip install -e pyroms/pyroms_toolbox
$ pip install -e pyroms/bathy_smoother
``` 
   
## Install scrip

If you install as above and try to import the three Pyroms modules without having installed
scrip you will get a warning like this:

```
$ python
Python 3.8.5 | packaged by conda-forge | (default, Aug 29 2020, 01:22:49)
[GCC 7.5.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import pyroms
WARNING:root: scrip could not be imported. Remapping functions will not be available
>>> import pyroms_toolbox
>>> import bathy_smoother
```

The scrip module is not available via Conda or any other package repository and we are looking at alternatives. In the meantime, scrip can be built and installed from source as follows

```
# Start in the directory into which you cloned pyroms and cd to the SCRIP
# source directory
$ cd pyroms/pyroms/external/scrip/source/

# Print the location of the active Conda environment (which is called "python38"
# in this case). The active environment location is used to find the netCDF and
# other libraries.
$ conda info | grep "active env location"
    active env location : /home/hadfield/miniconda3/envs/python38

# Run make to build the scrip Python extension and install it into the Conda
# environment. The makefile calculates a variable called SCRIP_EXT_DIR, into
# which it installs the scrip Python extension. If pyroms has been installed
# in editable (development) mode, set the DEVELOP variable to a non-empty value.
$ export PREFIX=/home/hadfield/miniconda3/envs/python38
$ make DEVELOP=1 PREFIX=$PREFIX install
$ mv -vf scrip*.so ../../../pyroms
¡®scrip.cpython-38-x86_64-linux-gnu.so¡¯ -> ¡®../../../pyroms/scrip.cpython-38-x86_64-linux-gnu.so¡¯
```

## Removal

To remove the three Pyroms packages you can use the "pip uninstall" command, referring to the packages by their package names

```
# Run from any directory in the same environment as you installed
# and use the package name
$ pip uninstall pyroms
$ pip uninstall pyroms_toolbox
$ pip uninstall bathy_smoother
```  
   
	 