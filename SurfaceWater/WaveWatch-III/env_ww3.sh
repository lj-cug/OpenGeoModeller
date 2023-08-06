############# netcdf #################
export NETCDF_DIR=$installpath/netcdf
export NETCDF_LIBDIR=$NETCDF_DIR/lib
export NETCDF_INCDIR=$NETCDF_DIR/include
export PATH=$NETCDF_DIR/bin:$PATH
export LD_LIBRARY_PATH=$NETCDF_LIBDIR:$LD_LIBRARY_PATH
export WWATCH3_NETCDF=NC4
export NETCDF_CONFIG=$NETCDF_DIR/bin/nc-config

############# metis #################
export METIS_PATH=$installpath/parmetis-4.0.3/metis
export PATH=$METIS_PATH/bin:$PATH
export LD_LIBRARY_PATH=$METIS_PATH/lib:$LD_LIBRARY_PATH
