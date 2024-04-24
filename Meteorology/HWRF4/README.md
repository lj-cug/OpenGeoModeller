# HWRF4

飓风跟踪的数值气象模式，基于WRF框架

驱动数据主要是北美的再分析数据

## Critical Update On Reduced HWRF V4.0a Capabilities
The Hurricane WRF v4.0a public release will have reduced capabilities due to format and library changes impacting input datasets. 

Starting in January 2021, an update was made to the bufrlib version, impacting input data for data assimilation. 
Additionally, when GFSv16 becomes operational (expected March 2021), output will be in NetCDF format, which is not supported in HWRFv4.0a. Only use of GRIB2 data will be supported.
Implications of these changes are that HWRFv4.0a cannot run with data assimilation or vortex initialization. 
Historical cases may still be run using legacy input data formats.

Please check Known Issues and Fixes for more information. 

## HWRF Components And Libraries
The HWRF system consists of runtime scripts and source code for the HWRF Utilities, MPIPOM-TC, NCEP Coupler, GFDL Vortex Tracker, WRF, WPS, GSI, and UPP. Customized versions of all of these components are available from this site.

Additional libraries, as described in the HWRF Users Guide, are required.
