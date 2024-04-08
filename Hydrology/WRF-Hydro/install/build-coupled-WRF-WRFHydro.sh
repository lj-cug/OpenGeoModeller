# Download the source code for WRF
wget https://github.com/wrf-model/WRF/archive/v4.1.2.tar.gz
tar -xzvf *.tar.gz
mv WRF* WRF
rm *.tar.gz

# Download the source code for WPS
wget https://github.com/wrf-model/WPS/archive/v4.1.tar.gz
tar -xzvf *.tar.gz
mv WPS* WPS
rm *.tar.gz

# Download the source code for WRF-Hydro
wget https://github.com/NCAR/wrf_hydro_nwm_public/archive/v5.1.1.tar.gz
tar -xzvf *.tar.gz
rm *.tar.gz

# Replace the old version of WRF-Hydro distributed with WRF with updated WRF-Hydro source code
rm -r WRF/hydro
cp -r wrf_hydro*/trunk/NDHMS WRF/hydro

# 2 Ensure that any required dependencies are available on your system and 
#   that proper environment variables for paths and compile time options are set
source WRF/hydro/template/setEnvar.sh

# 3 Configure and compile WRF with WRF-Hydro
## Note that the WRF_HYDRO environment variable (included in the setEnvar.sh script) is what triggers a coupled compile. 
## To recompile WRF as standalone model set WRF_HYDRO to 0.

## Move to the WRF base directory
cd WRF

## Run configure and select appropriate options
./configure

## Select the dmpar option for your compiler
## Select option 1 - basic nesting
## Run compile and once complete check the log file for errors and warnings
./compile em_real >& compile.log

# 4 Configure and compile WPS
## Move to the WPS base directory
cd ../WPS

## Run configure and select appropriate options
./configure

## Select the option for your compiler
## Run compile and once complete check the log file for errors and warnings
./compile >& compile.log

# Running Coupled WRF | WRF-Hydro
## 1. Ensure the necessary files are available from previous steps in the WPS and WRF run process
## Check that the required wrfinput_d0x and wrfbdy_d01 (initial and boundary condition files) for your simulation have been created and are available in the run directory

## 2. Add required files for WRF-Hydro to the appropriate locations
## Move to the WRF run directory
cd ../WRF/run

## Create a DOMAIN directory for WRF-Hydro domain files
mkdir DOMAIN
## Move all the necessary WRF-Hydro domain files (e.g. Fulldom_hires.nc, Route_Link.nc, etc.) for your configuration of WRF-Hydro to the DOMAIN directory
## Copy the template WRF-Hydro hydro.namelist from the WRF-Hydro source code and 
## any necessary *.TBL files (e.g. CHANPARM.TBL) for your configuration into the run directory. 
## Note that the land surface model parameter files (e.g. SOILPARM.TBL, MPTABLE.TBL) are already distributed with WRF.

## 3. Edit the WRF and WRF-Hydro namelists as needed
## 4. Run the coupled model
mpirun -np 4 ./wrf.exe >& wrf_coupled.log
