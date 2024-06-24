# wrf_hydro项目介绍
https://ral.ucar.edu/projects/wrf_hydro/documentation

# wrf_hydro_v5.2 (nmw_v2.1)
```
Below are some highlights. Version 5.2.0 of the Community WRF-Hydro source code is consistent with the NOAA National Water Model (NWM) v2.1 code in operations plus the following additions and fixes:

Added snow-related vars to retrospective output configuration
Added SFCFRNOFF output flag tied to overland runoff switch
Support GNU Fortran version 10.x and 11.x
Added missing crop and soil NoahMP namelist options
Removed deprecated namelist options for Reach-Lakes configuration
Fixed issues with using gridded domains with very small dimensions
Added ISLAKE mask flag to relevant NoahMP code
Updated NUOPC cap component with improved MPI support and to read and write restarts
Updated default channel parameters for gridded configuration
Performance improvements to channel network initialization
Added global title attribute to all output netCDF files
Updated namelist templates
```

# wrf_hydro代码开发仓库
https://github.com/NCAR/wrf_hydro_nwm_public

WRF-Hydro is a community modeling system and framework for hydrologic modeling and model coupling. 
In 2016 a configuration of WRF-Hydro was implemented as the National Water Model (NWM) for the continental United States.

## 构建wrf-hydro
https://github.com/NCAR/wrf_hydro_nwm_public/blob/main/docs/BUILD.md

# 前处理
```
https://ral.ucar.edu/projects/wrf_hydro/pre-processing-tools
https://github.com/NCAR/wrf_hydro_arcgis_preprocessor
https://ral.ucar.edu/projects/wrf_hydro/pre-processing-tools#preprocessing1
```

## 气象驱动数据
```
GFS
GLDAS
WRF
CMFD (China)
```
# 后处理
wrfhydro-usage

```
Rwrfhydro
xarray-Python
```

# 培训
https://github.com/NCAR/wrf_hydro_training

# FAQ's
https://ral.ucar.edu/projects/wrf_hydro/faqs

# WRF-Hydro V5.2 Test Cases
```
Oahu Hawaii v5.2.x Test Case (NWM configuration only)
WRF-Hydro V5.2.x
Standalone Croton New York v5.2 Test Case
WRF-Hydro V5.2.0 User Guide (v5.1.1)
Coupled WRF|WRF-Hydro v5.2 Front Range Colorado Test Case
WRF-Hydro V5.2.0 User Guide (v5.1.1)
```