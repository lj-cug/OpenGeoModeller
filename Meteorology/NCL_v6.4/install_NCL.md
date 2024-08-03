# ncl_install
## conda
```
conda create -n ncl_stable -c conda-forge ncl
source activate ncl_stable
```

### Test NCL
```
ncl -V
ncl $NCARG_ROOT/lib/ncarg/nclex/gsun/gsun02n.ncl
ncl $NCARG_ROOT/lib/ncarg/nclex/nug/NUG_multi_timeseries.ncl
```

### update
conda update -n ncl_stable -c conda-forge --all

### check NCL
conda list -n ncl_stable
conda --version
which ncl
ncl -V
env  | grep NCARG

## Installing NCL from a precompiled binary
```
sudo apt-get install ncl-ncarg
sudo ln -s /usr/share/ncarg /usr/lib/ncarg
```
## buid from source code
https://www.ncl.ucar.edu/Download/build_from_src.shtml

