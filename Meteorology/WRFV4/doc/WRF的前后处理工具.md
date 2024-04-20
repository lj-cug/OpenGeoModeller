# WRF的前后处理工具
## ARWpost与GrADS
ARWpost是一个把WRF 结果转为GrADS 或Vis5D 可以辨识的数据格式的软件.

## Rwrf (R脚本)
Installing, Simulating, and Visualizing WRF with R, used for installing, running, and visualizing WRF with R Programming Language.

## Python脚本工具
[前后处理及自动运行](./Python脚本工具.md)

## NCL脚本工具
### PostWRF
Amirhossein Nikfal. 2023. PostWRF: Interactive tools for the visualization of the WRF and ERA5 model outputs. Environmental Modelling and Software,160: 105591

### NCL-Scripts-for-WRF
This repository includes NCL scripts that can be used to post-processing WRF outs, including but not limited to spatial plots, write WRF outputs to csv files, PBL height calculation based on Richardson method, and time-height plots. Please feel free to contact Xia Sun (emsunxia@gmail.com) if you have any questions. I would happy to help.

### pih-ncl-scripts
NCL scripts to create WRF graphics.
The scripts herein are for post-processed WRF graphics. You will need to modify the NCL scripts
to reflect the location of your WRF NetCDF output files and domain, etc.

### NCL_vortex_tracker
涡旋跟踪,未测试...

## GeoTIFF2geogrid和geogrid2GeoTIFF转换
模拟WRF-fire, 网格尺寸小于10m, 需要使用高分辨率的GeoTIFF格式的静态数据《
因此需要转换.

Jonathan D. Beezley. Integrating high-resolution static data into WRF for real fire simulations.

## gis4wrf-0.14.7
基于QGIS的WRF前处理插件

D. Meyera, M. Riechert. 2019. Open source QGIS toolkit for the Advanced Research WRF modelling system. Environmental Modelling and Software,112: 166-178.

## Tecplot和ParaView可视化
tecio_wrf.dll (仅Windows OS可用)
