# wrf_hydro项目介绍

[WRF-Hydro documentation](https://ral.ucar.edu/projects/wrf_hydro/documentation)

## 本仓库文档

- [install](./install/)：独立 / 耦合编译脚本
- [doc](./doc/)：前处理、测试算例与驱动数据
- [rwrfhydro](./rwrfhydro/)：R 后处理

## wrf_hydro_v5.2 (nmw_v2.1)
```text
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

## wrf_hydro代码开发仓库
https://github.com/NCAR/wrf_hydro_nwm_public

WRF-Hydro is a community modeling system and framework for hydrologic modeling and model coupling. 
In 2016 a configuration of WRF-Hydro was implemented as the National Water Model (NWM) for the continental United States.

### 构建wrf-hydro
https://github.com/NCAR/wrf_hydro_nwm_public/blob/main/docs/BUILD.md

## 前处理
```text
https://ral.ucar.edu/projects/wrf_hydro/pre-processing-tools
https://github.com/NCAR/wrf_hydro_arcgis_preprocessor
https://ral.ucar.edu/projects/wrf_hydro/pre-processing-tools#preprocessing1
```

详见 [doc/前处理工具.md](./doc/前处理工具.md)

### 气象驱动数据
```text
GFS
GLDAS
WRF
CMFD (China)
```
## 后处理
wrfhydro-usage

```text
Rwrfhydro
xarray-Python
```

详见 [rwrfhydro](./rwrfhydro/)

## 培训
https://github.com/NCAR/wrf_hydro_training

## FAQ's
https://ral.ucar.edu/projects/wrf_hydro/faqs

## WRF-Hydro V5.2 Test Cases
```text
Oahu Hawaii v5.2.x Test Case (NWM configuration only)
WRF-Hydro V5.2.x
Standalone Croton New York v5.2 Test Case
WRF-Hydro V5.2.0 User Guide (v5.1.1)
Coupled WRF|WRF-Hydro v5.2 Front Range Colorado Test Case
WRF-Hydro V5.2.0 User Guide (v5.1.1)
```

详见 [doc/测试算例.md](./doc/测试算例.md)

## 相关文档

- [Hydrology](../)：学科入口
- [Meteorology/WRFV4](../../Meteorology/WRFV4/)：耦合大气模式 WRF
- [install/README.md](./install/README.md)：编译注意（NetCDF-Fortran）
- [hpc-base](../../hpc-base/)：编译器、MPI、NetCDF 等
