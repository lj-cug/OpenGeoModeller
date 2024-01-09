# HDF5 bashrc
```
# HDF5-related
export VOL_DIR=/home/lijian/HDF5/vol-async
export ABT_DIR=$VOL_DIR/argobots
export H5_DIR=/home/lijian/HDF5/hdf5

export LD_LIBRARY_PATH=$VOL_DIR/src:$H5_DIR/install/lib:$ABT_DIR/install/lib:$LD_LIBRARY_PATH
export HDF5_PLUGIN_PATH="$VOL_DIR/src"
# for vol_async
export HDF5_VOL_CONNECTOR="async under_vol=0;under_info={}"
# for vol_cache
#export HDF5_VOL_CONNECTOR="cache_ext config=cache_1.cfg;under_vol=512;under_info={under_vol=0;under_info={}}"

```