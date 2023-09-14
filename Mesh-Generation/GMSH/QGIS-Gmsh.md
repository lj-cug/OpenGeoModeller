# QGIS-Gmsh

生成[GMSH网格生成器](http://geuz.org/gmsh)的几何输入文件，然后转换GMSH网格文件为shapefiles，可以导入到QGIS

## 安装

在 "Plugins>Manage and Install Plugins..." 中输入GMSH即可自动安装.

## 使用

https://github.com/jonathanlambrechts/qgis-gmsh/wiki

### 海岸线(水陆边界线)

计算域边界是QGIS的vector layers。
全球海岸线可以从[GSHHG](http://www.soest.hawaii.edu/pwessel/gshhg/)获取。

使用数据：http://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.4.zip

http://youtube.com/embed/qu67lC54B_8

### Shelf break

水下地形从[ETOPO1数据库](https://www.ngdc.noaa.gov/mgg/global/global.html)提取。

使用NetCDF文件：https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/grid_registered/netcdf/ETOPO1_Ice_g_gmt4.grd.gz

shelf break选则200m等深线，使用QGIS获得.

http://youtube.com/embed/AMK4SkkNlJQ

### 外部边界

We close the domain on the northern and southern boundary. The plugin will detect duplicated points if they have the exact same position. To ensure this, the QGIS snapping options are enabled. Different physical tags (in this case north, south, coast, shelf, and islands) can be assigned to the different lines by adding a properties physical to those features. Those tags will be transfered to the final mesh and can be used to define the boundary conditions.

http://youtube.com/embed/pDYHYZO-aNE

### 网格划分

Now we define a raster layer to specify the mesh size. In this case, we chose element sizes proportional to the square root of the bathymetry l = 300 * sqrt(max(-h, 1)), which in the QGIS raster calculator language reads:

300 * sqrt(("bathymetry@1" > -1) * 1  + ("bathymetry@1" <= -1) *  (-"bathymetry@1")).
The mesh projection (in this case UTM zone 55 south) is determined by the projection of the mesh size layer. The unit of the element sizes is the unit of the projected space (in this case meters).

Finally, the qgis-gmsh plugin is used to:

generate the geometry (.geo) input file for GMSH,
call GMSH to generate the mesh,
convert the resulting mesh to shape files and load it into QGIS.

http://youtube.com/embed/_ufIPIBLiMU

## 参考文献

QGIS Geographic Information System. Open Source Geospatial Foundation Project. http://qgis.osgeo.org

Gmsh: a three-dimensional finite element mesh generator with built-in pre- and post-processing facilities. C. Geuzaine and J.-F. Remacle. International Journal for Numerical Methods in Engineering 79, 1309-1331, 2009. http://geuz.org/gmsh

Multiscale mesh generation on the sphere. J. Lambrechts, R. Comblen, V. Legat, C. Geuzaine and J.-F. Remacle. Ocean Dynamics 58, 461-473, 2008.