## 13、NetCDF数据文件的处理

NetCDF(network Common Data
Form)网络通用数据格式是由美国大学大气研究协会（University Corporation
for Atmospheric
Research，UCAR)的Unidata项目科学家针对科学数据的特点开发的，是一种面向数组型并适于网络共享的数据的描述和编码标准。由于其灵活性，能够传输海量的面向阵列（array-oriented）数据，目前广泛应用于大气科学、水文、海洋学、环境模拟、地球物理等诸多领域，例如，NCEP(美国国家环境预报中心)发布的再分析资料，NOAA的CDC(气候数据中心)发布的海洋与大气综合数据集(COADS)均采用NetCDF作为标准。

ArcGIS软件：

（1）在ArcGIS中打开时，使用工具箱工具：Multidimension Tools---Make
NetCDFRaster Layer。这样就可以在ArcGIS中打开相应的数据了。

（2）作为一种方便的多维数据格式，NetCDF在存储时可以选择设置不同数据作为其维度。在这里是以时间为维度划分的。所以要先在Layer
Properties 的NetCDF选项卡中设置Band
Dimension。这里选择Time作为维度波段。

（3）确定后再次打开Layer
Properties，Symbology选项卡中会出现Band这一行选项，在后面的下来列表中就可以选择相应日期或者月份的波段进行显示。比如一年的daily数据，相应的就会有365个波段。

（4）如果需要对某一天的数据进行处理，则可以使用Export
data输出成栅格数据，然后就可以进行每天单波段数据的处理了。

MATLAB：2011b以上版本

1、打开netCDF数据文件：

ncid = netcdf.open(filename, mode)   mode:\'NC_WRITE\',\'NC_SHARE\',\'NC_NOWRITE\' 

2、关闭netCDF 文件：netcdf.close(ncid)

FORTRAN：参考一个例子程序，一般需要以下步骤的操作：

nf90_create：创建nc文件

nf90_def_dim：定义nc文件的维度

nf90_def_var：创建nc的元数据变量

nf90_enddef：完成创建nc元数据

nf90_put_var：将变量写入nc文件

nf90_close：关闭nc文件
