#
cd trunk/NDHMS
cp template/setEnvar.sh .

# Edit the WRF-Hydro environment variables (compile time options) in the file setEnvar.sh

# set NetCDF Environment
export NETCDF=`nc-config --prefix`
## If nc-config is not in your path then, export the following environment variables.
export NETCDF_INC="/netCdfPathOnYourMachine/include"
export NETCDF_LIB="/netCdfPathOnYourMachine/lib"
## Note that we do not currently support parallel netCDF.

# Run the configure script
./configure

## To compile WRF-Hydro offline with Noah land surface model, the following command is used:
./compile_offline_Noah.sh setEnvar.sh

## To compile WRF-Hydro offline with the NoahMP land surface model, the following script is used:
./compile_offline_NoahMP.sh setEnvar.sh

make clean
make

# If make (compilation) is successful, the executable file created for the uncoupled WRF-Hydro model is created for NoahMP in the Run/ directory:
wrf_hydro.exe

# Then, in Run/ , wrf_hydro.exe is copied to wrf_hydro_NoahMP.exe and then
# wrf_hydro_NoahMP.exe is symlinked to wrf_hydro.exe.
# Note: if running make after previously running the compile_offline_NoahMP.sh script, only the
# wrf_hydro.exe file is updated. To avoid confusion use a make clean before each compile.
# Finally, the environment variables used in the compile are printed.
