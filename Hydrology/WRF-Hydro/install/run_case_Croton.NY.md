# Run test case
## oahu-hi test case
Simple case.

## Croton, NY test case
This walkthrough is using the Croton, NY example test case. 

Details on the domain and time period of the simulation are provided in the example_case/Readme.txt file.

```
For the WRF-Hydro Gridded configuration, these files are located in the directory example_case/Gridded.

# 1.1 Copy the *.TBL files to the Gridded directory, for example:
cp wrf_hydro_nwm_public*/trunk/NDHMS/Run/*.TBL example_case/Gridded

# 1.2 Copy the wrf_hydro.exe file to the Gridded directory
cp wrf_hydro_nwm_public*/trunk/NDHMS/Run/wrf_hydro.exe example_case/Gridded

# 2 Next we need to copy our forcing data to the Gridded directory.
cp -r ../FORCING .

# 3 Now we will run the simulation. Note that there are many options and filepaths that need to be set in the
two namelist files ¡°hydro.namelist¡± and ¡°namelist.hrldas¡±.

## Before running the model, ensure you are in the example_case/Gridded directory.
## We will now run the model using mpirun with 2 cores.
mpirun -np 2 ./wrf_hydro.exe

# 4 If your simulation ran successfully, there should now be a large number of output files.
## There are also two important files for determining the success
or failure of the run, ¡°diag_hydro.00000¡± and ¡°diag_hydro.00001¡±.

cat diag_hydro.00000

If this line is not present, the simulation did not finish successfully.

# 5. You can check the validity of your simulation results by comparing the restart files produced during your
model run with the restart files included in the example_case/Gridded/referenceSim directory.

Note: Our current example test case has only been run and tested with the Noah-MP land surface model. For
information regarding running WRF-Hydro with Noah please see the WRF-Hydro V5 Technical Description.
```

## Coupled WRF-WRF-Hydro test case
front_range_CO_example_testcase_coupled

