# rwrfhydro-API
简要列出rwrfhydsro软件包中vignettes的用法。还可以参考WRFHydro_PostProcessing的R脚本。
## Download and process MODIS LAI images
We are using WRF-Hydro to predict streamflow for Fourmile Creek at the Orodell USGS gage for the 2013 snowmelt period. We want to obtain MODIS LAI images for our region of interest and insert the processed images into our forcing time series so we can use dynamic LAI in future WRF-Hydro model runs.
### Setup the MODIS-R tool

### Download and process MODIS tiles
Here we download the relevant MODIS tiles and resample to the Fourmile Creek geogrid. The GetMODIS tool calls the MODIS-R runGdal tool to identify the required MODIS tiles to cover the domain, download the appropriate product tiles, mosaic them, and clip/resample (nearest neighbor) to the geogrid extent and resolution. The result will be a directory of processed TIF images in the outDirPath specified in MODISoptions. We specify the geogrid, a start and end date, and the MOD15A2 (FPAR/LAI) MODIS product name.

### Apply smoothing filters
Apply a whittaker filter to smooth the LAI time series. This is generally not necessary for a derived product like LAI, but may be useful for products like NDVI, snow cover fraction, vegetated fraction, etc.

### Export to forcing
Export the second smoothed time series to the forcing data for use in future model runs. This requires a local NCO installation.

## Spatial tools  空间分析工具(GIS)
```
GetProj
GetGeogridSpatialInfo
ExportGeogrid
GetGeogridIndex
GetTimeZone
GetRfc
GetPoly
PolygonToRaster
```
### General Info
```
library(rwrfhydro)
library(rgdal)
options(warn=1)
fcPath <- '~/wrfHydroTestCases/Fourmile_Creek_testcase_v2.0'
geoFile <- paste0(fcPath,'/DOMAIN/geo_em_d01.Fourmile1km.nlcd11.nc')
```
### GetProj
```
proj4 <- GetProj(geoFile)
proj4
```
### GetGeogridSpatialInfo
```
geoInfo <- GetGeogridSpatialInfo(geoFile)
geoInfo
```
### ExportGeogrid
```
head(ncdump(geoFile))
ExportGeogrid(geoFile,"HGT_M", "geogrid_hgt.tif")

# Read the newly created tiff file
library(raster)
r <- raster("geogrid_hgt.tif")

# Plot the imported raster from tiff file
plot(r, main = "HGT_M", col=terrain.colors(100))

# Check the raser information and notice that geographic coordinate information has been added.
r

file <- paste0(fcPath,"/run.FluxEval/RESTART.2013060100_DOMAIN1")
# ncdump(file) # check if the SOIL_T exist in the file

# Export the 3rd layer of the 4-layer soil temperature variable
ExportGeogrid(file,
             inVar="SOIL_T",
             outFile="20130315_soilm3.tif",
             inCoordFile=geoFile,
             inLyr=3)

# Read the newly created tiff file
r <- raster("20130315_soilm3.tif")

# Plot the imported raster from tiff file
plot(r, main = "Soil Temperature", col=rev(heat.colors(100))) # in raster
```
### GetGeogridIndex
### GetTimeZone
### GetRfc
### GetPoly
### PolyToRaster

## Get data from multiple netcdf files
```
library("rwrfhydro")
dataPath <- '~/wrfHydroTestCases/Fourmile_Creek_testcase_v2.0/run.FullRouting/'
```
### List-based data retrieval

### Plot the timeseries


## Domain and Channel Visualization  流域和河道的可视化
```
library("rwrfhydro")
library(rgdal)  ## on some linux machines this appears needed
fcPath <- '~/wrfHydroTestCases/Fourmile_Creek_testcase_v2.0'  ## Set the path to the directory of WRF Hydro test cases
```
### Visualize Gridded Domain Data: VisualizeDomain()  可视化网格化流域数据
(1) Land surface model pixels
```
geoFile <- paste0(fcPath,'/run.ChannelRouting/DOMAIN/geo_em_d01.Fourmile1km.nlcd11.nc')
coordsProj <- GetDomainCoordsProj(geoFile)
ClosureGeo <- VisualizeDomain(geoFile, plotVar='HGT_M', plot=FALSE, plotDf=coordsProj)
```
(2) Routing pixels
```
......
```
### VisualizeChanNtwk()  可视化河网数据
```
chrtFile <- paste0(fcPath,'/run.FullRouting/201306010000.CHRTOUT_DOMAIN1')
LocLinkFun<- VisualizeChanNtwk(chrtFile)
```

## Evaluate evapotranspiration simulation   评估蒸散发
```
library("rwrfhydro")
dataPath <- '~/wrfHydroTestCases/Fourmile_Creek_testcase_v2.0/'
```
### Import modelled datasets  导入模拟数据集
### Import observed datasets  导入观测数据集
### Plot & compare the time series of ET fluxes   蒸散发通量
### Review model performance statistics   评估模型精度
### Review hourly fluxes for May  小时尺度的通量时间序列过程
### Plot & compare the time series of energy fluxes 能量通量
### Plot & compare the time series of turbulent fluxes 湍流通量

## Precipitation Evaluation   评估降雨
### Import observed datasets
### Import forcing/precipitation data used in WRF-Hydro model
```
Aggregating hourly data into daily.
Comparing daily QPE/QPF versus GHCN-D
Calculate statistics over RFCs
Calculate statistics over polygons
Calculate categorical statistics
```
## Collect USGS stream observations to evaluate streamflow simulation   河道径流观测与模拟的对比
### Discover gage locations, get data, and save to local database
### Query the local data
### Import modelled and observed datasets
### Plot hydrographs
### Review flow duration curves
### Review model performance statistics
### GagesII Attributes

## Evaluate water budget partitioning with rwrfhydro  水量收支平衡分析
### Import modelled datasets
```
library(doMC)
registerDoMC(3)
modLdasoutWb1h.allrt.fc <- ReadLdasoutWb(paste0(dataPath, '/run.FullRouting'), 
                                         paste0(dataPath, '/DOMAIN/Fulldom_hires_hydrofile.Fourmile100m.nc'), 
                                         mskvar="basn_msk", basid=1, aggfact=10, parallel=TRUE)
										 
# Calculate basin-averaged routing water fluxes.
modRtout1h.allrt.fc <- ReadRtout(paste0(dataPath, '/run.FullRouting'), 
                                 paste0(dataPath, '/DOMAIN/Fulldom_hires_hydrofile.Fourmile100m.nc'), 
                                 mskvar="basn_msk", basid=1, parallel=TRUE)
								 
# Import groundwater outflow model output.
modGwout.allrt.fc <- ReadGwOut(paste0(dataPath, '/run.FullRouting/GW_outflow.txt'))										 
```
### Evaluate the predicted water budget
```
wb.allrt.fc <- CalcNoahmpWatBudg(modLdasoutWb1h.allrt.fc, rtoutDf=modRtout1h.allrt.fc, 
                                 gwoutDf=modGwout.allrt.fc, sfcrt=TRUE, basarea=63.1)
wb.allrt.fc
PlotWatBudg(wb.allrt.fc)
PlotWatBudg(wb.allrt.fc, "bar")								 
```
## Collect the SNODAS product and build a local database
Snow Data Assimilation System (SNODAS) Data Products

## Collect SNOTEL SWE from Niwot Ridge and compare directly to model output
Snow Telemetry (SNOTEL) Network
