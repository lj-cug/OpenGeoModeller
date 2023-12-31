# 1、PIHMGis_v2.3安装

（1）Windows系统，下载PIHMGis_V2.3安装程序。

![](./media/image1.png)

（2）启动QGIS软件，使用Launch_PIHMgis_v2.3.cmd

（3）激活PIHMgis工具条，在QGIS的Manage
Plugins\...中激活。将出现下面界面：

![](./media/image2.png)

![](./media/image3.png)

# 2、PIHMGis操作步骤

\(1\) **Raster Processing**

需要7步。

![](./media/image4.png)

按照V_Catchment的例子来操作。

加载DEM数据：

***Layers \>\> Add Raster Layer***

QGIS软件可以读取ESRI binary (.adf)或ArcInfo的ASCII
(.asc)两种格式的DEM原始数据。

然后，照上面的顺序逐个计算栅格文件。

在计算Catchment polygon时，首先用ArcGIS将Catchment
Grid的栅格数据为0的部分，替换为Nodata:
使用setnull函数，然后，conversion\--\>raster to polygon

\(2\) **Vector Processing**

![](./media/image5.png)

1\) Dissolve Polygon

使用QGIS软件进行此操作总是出错，于是，使用ArcGIS软件：Data Management
Tools \-- \> Features \--\> Polygon to line

2\) Polygon To Line

由此，ArcGIS软件，实际上完成了QGIS软件的1步和2步。

生成：polyline.shp

3\) Simplify Line

![](./media/image6.png)

此步，需要导入polyline.shp和Raster Processing中生成的streamline.shp

4\) Split Line

![](./media/image7.png)

此步，需要导入Simplify Line生成的2个shapfile文件。

到此，需要进行了是：将河道延长至流域出口边界，具体做法是：

**File \--\> Project Properties \--\> Snapping options \--\> General**

![](./media/image8.png)

5\) Vector Merge

![](./media/image9.png)

**(3) Domain Decomposition**

![](./media/image10.png)

1\) Read ShapeTopology

![](./media/image11.png)

2\) Run TRIANGLE

使用开源软件TRIANGLE划分三角形单元。

![](./media/image12.png)

使用最大三角形单元面积：2000000m^2^，生成三角形单元207152个，节点数：105756。

![C:\\Users\\LI\\AppData\\Roaming\\Tencent\\Users\\94207625\\QQ\\WinTemp\\RichOle\\C}F\_\_\`\`Y)AC_A4JCCDUM%I6.png](./media/image13.png)

**(4) DataModel Loader**

![](./media/image14.png)

1\) Mesh File

![](./media/image15.png)
