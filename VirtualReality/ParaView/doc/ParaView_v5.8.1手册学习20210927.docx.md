# ParaView_v5.8.1手册学习

# 第1章 引言

可视化基本知识：

基于数据流(data flow)，经过算法处理（如clipping,
slicing,），转换为图片。

![](./media/image1.emf){width="4.43084864391951in"
height="1.2093897637795274in"}

图1.1可视化模型：处理对象A，B，C（源、过滤和映射对象: [source, filter,
mapper objects]{.mark}）

Reader: 读入数据(Source)

Pipeline

VTK model

## 可执行程序

paraview（客户端，基于Qt5）

pvpython：交互式Python脚本

pvbatch：批处理，可以并行运行。

pvserver: 远程可视化，远程获取HPC上的数据，在客户端PC可视化

pvdataserver

pvrenderserver

UI

Creating a source

Changing properties

Applying filters

## 使用pvpython

**from** paraview.simple **import** \*

[脚本行为的跟踪：]{.mark}paraview支持在UI中跟踪你的行为，使用python脚本。点击：Tools-\>Start
Trace；现在进入跟踪模式，当用户创建源、过滤、改变属性、点击Apply等等，所有动作都被监控；完成跟踪，点击Tools-\>Stop
Trace

[备注：]{.mark}Python脚本跟踪对初学者很重要，可以了解每个动作的Python脚本命令实施。

# 第2章 加载数据

File-\>Open（快捷键：Ctrl+O）

动画：VCR Controls ![](./media/image2.emf){width="1.0814873140857393in"
height="0.22808180227471567in"}

打开近期打开过的文件：File-\>Recent Files

命令行打开：paraview \--data =\.../ ParaViewData /Data/can.ex2

paraview \--data =\.../ ParaViewData /Data/my.vtk

## Reader的一般属性

PV使用不同的文件格式读取程序，每个都有不同的属性。reader的属性包括：

Selecting data arrays：允许选择加载数据的数组(cell-centered, point
centered,
其他)。有时，仅加载已知的数组来可视化，这可节省内存和处理时间。

## 在pvpython中打开数据文件

提供了OpenDataFile函数：

\>\>\> reader = OpenDataFile (\"\.../ ParaViewData /Data/can.ex2\")

\>\>\> if reader:

\... print(\"Success\")

\... else:

\... print(\"Failed\")

\...

\>\>\> reader = ExodusIIReader (FileName=\"\.../ ParaViewData
/Data/can.ex2\")

## 重新加载文件

File-\>Reload Files

Reload existing file(s)：加载已打开过的文件

Find new files：让reader知道任何新的文件。

# 第3章 了解数据

## 3.1VTK数据模型

VTK中最基本的数据结构是data
object：或者是科学数据，如矩形网格或FEM网格；或者是更抽象的数据结构，如graphy或tree

这些数据结构由更小的块构成：网格(拓扑和几何)和属性(attributes)。

### 3.1.1网格 {#网格 .标题3}

vertices(points)

cells(elements, zones)

### 3.1.2属性(fields, arrays) {#属性fields-arrays .标题3}

包括：压力、温度、速度和切应力等

pointe-centered: 在节点上定义变量，单元内的分布插值得到

cell-centered：假设整个单元内均匀分布。可以应用：Cell Data to Point
Data过滤，在PV中该过滤是自动执行的

### 3.1.3均匀矩形网格(图像数据) {#均匀矩形网格图像数据 .标题3}

Extents

Origin

Spacing

### 3.1.4矩形网格 {#矩形网格 .标题3}

Extents

Three arrays defining coordinates in the x-, y- and z-directions

### 3.1.5曲线网格（结构网格） {#曲线网格结构网格 .标题3}

Extents

An array of point coordinates - This array stores the position of each
vertex explicitly.

### 3.1.6AMR数据集 {#amr数据集 .标题3}

VTK支持Berger-Oliger类型的AMR数据集，如下图。

![](./media/image3.emf){width="2.8936734470691166in"
height="2.90755905511811in"}

### 3.1.7非结构网格 {#非结构网格 .标题3}

支持的非结构网格类型参考：vtkCellType.h

![](./media/image4.emf){width="4.390509623797025in"
height="1.0732261592300962in"}

![](./media/image5.emf){width="5.237066929133858in"
height="2.9328226159230097in"}

### 3.1.8多边形网格(polydata) {#多边形网格polydata .标题3}

### 3.1.9表 {#表 .标题3}

### 3.1.10多块数据 {#多块数据 .标题3}

## 3.2获取数据信息

### 3.2.1 Information面板 {#information面板 .标题3}

View-\>Information

Properties

Statistics

Data Arrays

Bounds

Time

### 3.2.2 Statistics Inspector面板 {#statistics-inspector面板 .标题3}

Geometry Size

## 3.3在pvpython中获取数据信息

# 第4章 显示数据

views：是sinks，获取输入数据，但不输出数据

[不同类型：]{.mark}

-   Rendering Views: 包括：Render Vies, Slice View, Quad View

-   Chart Views: 可视化非几何数据。包括：Line Chart View, Bar Chart
    View, Bag Chart View, Parallel Coordinates Views

-   Comparative Views：参数研究，可视化改变参数后对计算结果的影响。

## 4.1多个视图

### 4.1.1paraview中的多个视图 {#paraview中的多个视图 .标题3}

Split View

\+ X

### 4.1.2pvpython中的多个视图 {#pvpython中的多个视图 .标题3}

**from** paraview.simple **import** \*

\>\>\> view = CreateRenderView ()

\# Alternatively, use CreateView.

\>\>\> view = CreateView(\"RenderView\")

## 4.2视图(view)属性

Just like parameters on pipeline modules, such as readers and filters,
views provide parameters that can be used for customizing the
visualization such as changing the background color for rendering views
and adding title texts for chart views. These parameters are referred to
as View Properties and are accessible from the Properties panel in
paraview.

### 4.2.1paraview中的视图属性 {#paraview中的视图属性 .标题3}

Properties面板-\>View Apply

### 4.2.2pvpython中的视图属性 {#pvpython中的视图属性 .标题3}

\# 1. Save reference when a view is created

\>\>\> view = CreateView(\"RenderView\")

\# 2. Get reference to the active view.

\>\>\> view = GetActiveView ()

## 4.3显示(display)属性

Display properties refers to available parameters that control how data
from a pipeline module is displayed in a view, e.g., choosing to view
the output mesh as a wireframe, coloring the mesh using a data
attribute, and selecting which attributes to plot in chart view. A set
of display properties is associated with a particular pipeline module
and view. Thus, if the data output from a source is shown in two views,
there will be two sets of display properties used to control the
appearance of the data in each of the two views.

### 4.3.1在paraview中看显示属性 {#在paraview中看显示属性 .标题3}

无需apply。

### 4.2.2pvpython中看显示属性 {#pvpython中看显示属性 .标题3}

\# Using SetDisplayProperties/GetDisplayProperties to access the display

\# properties for the active source in the active view.

\>\>\> **print**( GetDisplayProperties (\"Opacity\"))

1.0

\>\>\> SetDisplayProperties (Opacity =0.5)

## 4.4 渲染视图（Render View）

Render View is the most commonly used view in ParaView. It is used to
render geometries and volumes in a 3D scene. This is the view that you
typically think of when referring to 3D visualization. The view relies
on techniques to map data to graphics primitives such as triangles,
polygons, and voxels, and it renders them in a scene.

Most of the scientific datasets discussed in Section 3.1 are comprised
of meshes. These meshes can be mapped to graphics primitives using
several of the established visualization techniques. (e.g., you can
compute the outer surface of these meshes and then render that surface
as filled polygons, you can just render the edges, or you can render the
data as a nebulous blob to get a better understanding of the internal
structure in the dataset.) Plugins, like [Surface
LIC]{.mark}（一个图形渲染的插件）, can provide additional ways of
rendering data using advanced techniques that provide more insight into
the data.

如果数据集不能表示为网格，如[表格]{.mark}，不能在渲染视图中显示。但这种情况，可以映射列数据，构建网格，然后构建点云。

[批注]{.mark}：渲染；Surface LIC插件？

### 4.4.1了解渲染过程 {#了解渲染过程 .标题3}

Surface rendering：实体表面的渲染。

Slice rendering：结构网格数据。

Volume rendering：Ray Tracing渲染。

显示属性，Properties面板或查看Representation Toolbar。

### 4.4.2 Render View in paraview {#render-view-in-paraview .标题3}

Creating a Render View：右击视图的标题栏，Convert
to子菜单，将[视图]{.mark}转换为渲染视图。

使用Pipeline Browser控制该视图管线模块产生的数据集的可视性。eyeball
icon反映[可视性状态]{.mark}。

Interactions：与渲染视图交互，移动视窗中的相机来探索可视化，以及设置最优视角。使用鼠标按键以及键盘修改键Ctrl，移动相机。可从在Setting对话框中的Camera标签改变交互模式：Tools-\>Settings。

PV中有[6种]{.mark}交互模式：Pan, Zoom, Roll, Rotate, Zoom to Mouse,
Multi Rotate

PV自动决定交互模式，根据加载的数据是2D还是3D。

### 4.4.3 Render View in pvpython {#render-view-in-pvpython .标题3}

创建render view

1 \>\>\> **from** paraview.simple **import** \*

2 \>\>\> view = CreateRenderView ()

3 \# Alternatively, use CreateView.

4 \>\>\> view = CreateView(\"RenderView\")

You use [Show and Hide]{.mark} to show or hide data produced by a
pipeline module in the view.

交互：因为pvpython是设计来执行脚本和批处理的，直接对视图的交互功能有限。为与视图交互，在Python中激活Interact函数：Interact
()

但更常用的是，编程[改变相机]{.mark}参数，如下：

1 \# Get camera from the active view, if possible.

2 \>\>\> camera = GetActiveCamera ()

3

4 \# or, get the camera from a specific render view.

5 \>\>\> camera = view. GetActiveCamera ()

6

7 \# Now, you can use methods on camera to move it around the scene.

8

9 \# Divide the camera's distance from the focal point by the given
dolly value.

10 \# Use a value greater than one to dolly-in toward the focal point,
and use a

11 \# value less than one to dolly-out away from the focal point.

12 \>\>\> camera.Dolly (10)

13

14 \# Set the roll angle of the camera about the direction of
projection.

15 \>\>\> camera.Roll (30)

16

17 \# Rotate the camera about the view up vector centered at the focal
point. Note

18 \# that the view up vector is whatever was set via SetViewUp, and is
not

19 \# necessarily perpendicular to the direction of projection. The
result is a

20 \# horizontal rotation of the camera.

21 \>\>\> camera.Azimuth (30)

22

23 \# Rotate the focal point about the view up vector, using the
camera's position

24 \# as the center of rotation. Note that the view up vector is
whatever was set

25 \# via SetViewUp, and is not necessarily perpendicular to the
direction of

26 \# projection. The result is a horizontal rotation of the scene.

27 \>\>\> camera.Yaw (10)

28

29 \# Rotate the camera about the cross product of the negative of the
direction

30 \# of projection and the view up vector, using the focal point as the
center

31 \# of rotation. The result is a vertical rotation of the scene.

32 \>\>\> camera.Elevation (10)

33

34 \# Rotate the focal point about the cross product of the view up
vector and the

35 \# direction of projection, using the camera's position as the center
of

36 \# rotation. The result is a vertical rotation of the camera.

37 \>\>\> camera.Pitch (10)

Alternatively, you can explicitly [set the camera position, focal point,
view up,]{.mark} etc,. to explicitly place the camera in the scene.

1 \>\>\> camera. SetFocalPoint (0, 0, 0)

2 \>\>\> camera.SetPosition (0, 0, -10)

3 \>\>\> camera.SetViewUp (0, 1, 0)

4 \>\>\> camera. SetViewAngle (30)

5 \>\>\> camera. SetParallelProjection (**False**)

6

7 \# If ParallelProjection is set to True, then you'll need

8 \# to specify parallel scalar as well i.e. the height of the viewport
in

9 \# world-coordinate distances. The default is 1. Note that the 'scale'

10 \# parameter works as an 'inverse scale' where larger numbers produce
smaller

11 \# images. This method has no effect in perspective projection mode.

12 \>\>\> camera. SetParallelScale (1)

View属性

1 \>\>\> view = GetActiveView ()

2

3 \# Set center axis visibility

4 \>\>\> view. CenterAxesVisibility = 0

5

6 \# Or you can use this variant to set the property on the active view.

7 \>\>\> SetViewProperties ( CenterAxesVisibility =0)

8

9 \# Another way of doing the same

10 \>\>\> SetViewProperties (view , CenterAxesVisibility =0)

11

12 \# Similarly, you can change orientation axes related properties

13 \>\>\> view. OrientationAxesVisibility = 0

14 \>\>\> view. OrientationAxesLabelColor = (1, 1, 1)

Display属性

Similar to view properties, display properties are accessible from the
display properties object or using the SetDisplayProperties function.

1 \>\>\> displayProperties = GetDisplayProperties (source , view)

2 \# Both source and view are optional. If not specified, the active
source

3 \# and active view will be used.

4

5 \# Now one can change properties on this object

6 \>\>\> displayProperties. Representation = \"Outline\"

7

8 \# Or use the SetDisplayProperties API.

9 \>\>\> SetDisplayProperties (source , view , Representation =Outline)

## 4.5 Line Chart视图

## 4.8 Plot Matrix视图

## 4.9 Parallel Coordinates视图

双重坐标轴。

## 4.10 Spreadsheet视图

## 4.11切片视图

## 4.12Python视图

## 4.13比较视图

Comparative Views, including Render View (Comparative), Line Chart View
(Comparative), and Bar Chart View (Comparative), are used for generating
comparative visualization from parameter studies.

参考第11章。

# 第5章 过滤数据

管线(pipelines)：source, filters, sinks (总称pipelines module or
algorithm)

数据流通过管线，在各节点上转移，直到形成图形，形成Sinks。

可视化过程就是启用各种可视化技术，如slicing, contouring, clipping, etc.,
都作为filters。

## 5.1了解过滤

A filter is a pipeline module with inputs and outputs. Data enters a
filter through the inputs. The filter transforms the data and produces
the resulting data on its outputs. A filter can have one or more input
and output ports. Each input port can optionally accept multiple input
connections.

![](./media/image6.emf){width="2.2336701662292215in"
height="1.328413167104112in"}

Figure 5.1 过滤

与readers类似，fliter属性允许用户控制过滤算法，可获取的属性与过滤自身有关。

## 5.2 在paraview中创建过滤

菜单FilterS，启用相关的过滤。

### 5.2.1 多个输入连接 {#多个输入连接 .标题3}

在Pipeline
Browser中，使用![](./media/image7.emf){width="1.347561242344707in"
height="0.20837270341207348in"}选择所有相关的管线模块，仅接受在输入端口上多个连接的在Filters菜单中启用。

### 5.2.2 多个输入端口 {#多个输入端口 .标题3}

[Figure 5.3:]{.mark} The Change Input Dialog is shown to allow you to
pick inputs for each of the input ports for a filter with multiple input
ports. To use this dialog, first select the Input Port you want to edit
on the left side, and select the pipeline module(s) that are to be
connected to this input port. Repeat the step for the other input
port(s). If an input port can accept multiple input connections, you can
select multiple modules, just like in the Pipeline Browser.

### 5.2.3 修改输入连接 {#修改输入连接 .标题3}

ParaView允许用户在创建过滤后修改[输入]{.mark}。To change inputs to a
filter, [right-click on the filter]{.mark} in the Pipeline Browser to
get the context menu, and then select [Change Input\....]{.mark} This
will pop up the same Change Input Dialog as when creating a filter with
multiple input ports.

[Figure 5.4:]{.mark} The context menu in the Pipeline Browser showing
the option to change inputs for a filter.

## 5.3在pvpython中创建过滤

使用其名称作为构造函数来创建对象：

1 \>\>\> from paraview.simple import \*

2 \...

3 \>\>\> filter = Shrink ()

过滤使用激活的源作为输入。另外，可显式定义输入作为函数形参：

1 \>\>\> reader = OpenDataFile (\...)

2 \...

3 \>\>\> shrink = Shift(Input=reader)

### 5.3.1 多个输入连接 {#多个输入连接-1 .标题3}

1 \>\>\> sphere = Sphere ()

2 \>\>\> cone = Cone ()

3

4 \# Simply pass the sources as a list to the constructor function.

5 \>\>\> appendDatasets = AppendDatasets (Input =\[ sphere , cone \])

6 \>\>\> **print**( appendDatasets .Input)

### 5.3.2 Multiple input ports {#multiple-input-ports .标题3}

1 \>\>\> sphere = Sphere ()

2 \>\>\> wavelet = Wavelet ()

3

4 \>\>\> resampleWithDataSet = ResampleWithDataset (Input=sphere,
Source=wavelet)

### 5.3.3 Changing input connections {#changing-input-connections .标题3}

1 \# For filter with single input connection

2 \>\>\> shrink.Input = cone

3

4 \# for filters with multiple input connects

5 \>\>\> appendDatasets .Input = \[reader , cone\]

6

7 \# to add a new input.

8 \>\>\> appendDatasets.Input.append(sphere)

9

10 \# to change multiple ports

11 \>\>\> resampleWithDataSet .Input = wavelet2

12 \>\>\> resampleWithDataSet .Source = cone

## 5.4 Changing filter properties in paraview

## 5.5 Changing filter properties in pvpython

使用pvpython可以获取过滤对象的属性，可以根据名称get或set其值，就像修改输入连接一样。

1 \# You can save the object reference when it's created.

2 \>\>\> shrink = Shrink ()

3

4 \# Or you can get access to the active source.

5 \>\>\> Shrink () \# \<\-- this will make the Shrink the active source.

6 \>\>\> shrink = GetActiveSource ()

7

8 \# To figure out available properties, you can always use help.

9 \>\>\> help(shrink)

10 Help on Shrink in module paraview. servermanager object:

## [5.6 Filters for sub-setting data]{.mark}

### [Clip]{.mark} {#clip .标题3}

![](./media/image8.emf){width="5.204089020122485in"
height="1.0913090551181102in"}

[Clip in pvpython]{.mark}

This following script demonstrates various aspects of using the Clip
filter in pvpython.

### [Slice]{.mark} {#slice .标题3}

![](./media/image9.emf){width="5.336432633420823in"
height="0.27405074365704285in"}

有Slice in Pvpython吗？

### 5.6.3 Extract Subset {#extract-subset .标题3}

Extract Subset in paraview

### Threshold {#threshold .标题3}

[Threshold in pvpython]{.mark}

### Iso Volume {#iso-volume .标题3}

有Iso Volume in pvpython吗？

### Extract Selection {#extract-selection .标题3}

## 5.7 Filters for geometric manipulation

### 5.7.1 Transform {#transform .标题3}

The Transform can be used to arbitrarily translate, rotate, and scale a
dataset. The transformation is applied by scaling the dataset, rotating
it, and then translating it based on the values specified.

### 5.7.2 Transform in paraview {#transform-in-paraview .标题3}

可以从Filters-\>Alphatical菜单中创建新的Transform。一旦创建好，就可以利用Properties面板设置转换，如：旋转、平移和尺度。

### 5.7.3 Transform in pvpython {#transform-in-pvpython .标题3}

1 \# To create the filter(if Input is not specified, the active source
will be

2 \# used as the input).

3 \>\>\> transform = Transform(Input =\...)

4

5 \# Set the transformation properties.

6 \>\>\> transform.Translate.Scale = \[1, 2, 1\]

7 \>\>\> transform.Transform.Translate = \[100 , 0, 0\]

8 \>\>\> transform.Transform.Rotate = \[0, 0, 0\]

### 5.7.4 Reflect {#reflect .标题3}

沿一个轴，做镜像。

## 5.8 Filters for sampling

这些过滤，计算新的数据集，表征数据集的一些必要特征，作为[输入]{.mark}（管线的输入）。

### 5.8.1 Glyph（字形） {#glyph字形 .标题3}

Glyphy用于放置标记或glphy在输入数据集的点位置。glphy可以旋转或调整大小，基于这些点的标量和矢量属性。

Filters或者![](./media/image10.emf){width="0.24285761154855642in"
height="0.2045330271216098in"}按钮，选择Glyph Type: Arrow, Sphere,
Cylinder, etc.

Figure 5.14: The Properties panel for the Glyph filter

glyph表征可提供更快速的渲染和低内存消耗，在生成3D几何体时需要，即输出glyph几何体到文件，需要glyph过滤。

### 5.8.2 Glyph With Custom Source {#glyph-with-custom-source .标题3}

Glyph With Custom Source与glyph一样，除了Glyph Type有限外。

Figure 5.16: Setting the Input and Glyph Source in the Glyph With Custom
Source filter.

### 5.8.3 Stream Tracer（流线） {#stream-tracer流线 .标题3}

The Stream Tracer filter is used to generate streamlines for vector
fields.产生流线：Filters-\>![](./media/image11.emf){width="0.2437937445319335in"
height="0.22652996500437445in"}

[几个控制参数]{.mark}：Integration Parameters，Integration
Direction，Integrator Type，Maximum Streamline Length

注意：因为Stream
Tracer过滤生成的是1D线的polydata，不能像Surface那样做阴影渲染（在Render
View）。为赋予流线3D结构，可对输出的流线做Tube过滤，Tube过滤的属性可以控制tube的粗度，可基于数据数组（即流线上采样点处的矢量场的幅度大小）改变管线粗细。

[Figure 5.18: The Properties panel showing the default properties for
the Stream Tracer filter.]{.mark}

使用Stream Tracer过滤的脚本程序：

1 \# find source

2 \>\>\> disk_out_refex2 = FindSource('disk_out_ref .ex2')

3

4 \# create a new 'Stream Tracer'

5 \>\>\> streamTracer1 = StreamTracer (Input=disk_out_refex2 ,
SeedType='Point Source ')

6

7 \>\>\> streamTracer1 .Vectors = \['POINTS ', 'V'\]

8

9 \# init the 'Point Source' selected for 'SeedType'

10 \>\>\> streamTracer1 .SeedType.Center = \[0.0 , 0.0,
0.07999992370605469\]

11 \>\>\> streamTracer1 .SeedType.Radius = 2.015999984741211

12

13 \# show data in view

14 \>\>\> Show ()

15

16 \# create a new 'Tube'

17 \>\>\> tube1 = Tube(Input= streamTracer1 )

18

19 \# Properties modified on tube1

20 \>\>\> tube1.Radius = 0.1611409378051758

21

22 \# show the data from tubes in view

23 \>\>\> Show ()

### 5.8.4 Stream Tracer With Custom Source {#stream-tracer-with-custom-source .标题3}

Stream Tracer allows you to specify the seed points either as a point
cloud or as a line source.

Figure 5.19: Streamlines generated from the disk out ref.ex2 dataset
using the output of the Slice filter as the Source for seed points.

### 5.8.5 Resample With Dataset {#resample-with-dataset .标题3}

在Filters菜单下

Resample With Dataset samples the point and cell attributes of one
dataset on to the points of another dataset

### 5.8.6 Resample To Image {#resample-to-image .标题3}

对均匀网格数据集实施效率更高。体渲染就是这样的操作。Resample to
Image过滤可以将任何数据集转换为Image数据。

### [5.8.7 Probe]{.mark} {#probe .标题3}

Probe可以在特殊点处采样数据集，获得单元数据属性以及插值点数据属性。

也可以使用SpreadSheet View或Information面板查看probed数据。

probe位置可以使用Render View中显示的交互式3D Widget定义。

### [5.8.8 Plot over line]{.mark} {#plot-over-line .标题3}

Plot Over Line沿着指定的线采样输入的数据集，然后在Line Chart
View绘制结果。与Probe过滤机制一样。

在Properties面板使用Resolution属性，控制沿线的采样点数目。

[批注：]{.mark}使用Start Trace跟踪Probe和Plot over
Line，获得Python脚本。可以尝试语音控制。

## 5.9 属性操作的过滤器

### 5.9.1 Calculator {#calculator .标题3}

### 5.9.2 Python calculator {#python-calculator .标题3}

### 5.9.3 Gradient {#gradient .标题3}

There are two filters that can compute gradients:

-   Gradient

-   Gradient of Unstructured DataSet

### 5.9.4 Mesh Quality {#mesh-quality .标题3}

The Mesh Quality filter creates a new cell array containing a geometric
measure of each cell's fitness.

-   Triangle Quality

-   Quad Quality

-   Tet Quality

-   HexQualityMeasure

## 5.10 White-box filters

Programmable Filter and Programmable Source

For these filters/sources, you can add Python code to do the data
generation or processing

## 5.11 Favorite filters

Filters-\>Favorites

Filters-\>Manage Favorites

## 5.12 最佳做法

### 5.12.1 避免数据爆炸 {#避免数据爆炸 .标题3}

when visualizing large datasets, it is important to understand the
memory requirements of filters.

remaining available memory is low. When you are not in danger of running
out of memory, the following advice is not relevant.

When dealing with structured data, it is absolutely important to know
what filters will change the data to unstructured. Unstructured data has
a much higher memory footprint, per cell, than structured data because
the topology must be explicitly written out. There are many filters in
ParaView that will change the topology in some way, and these filters
will write out the data as an unstructured grid, because that is the
only dataset that will handle any type of topology that is generated.
The following list of filters will write out a new unstructured topology
in its output that is roughly equivalent to the input. These filters
should never be used with structured data and should be used with
caution on unstructured data.

![](./media/image12.emf){width="5.140590551181102in"
height="1.0996609798775152in"}

# 第6章 选择数据

典型的可视化过程有2部分：设置可视化画面和实施结果分析（获取认知）。这个过程是迭代的。

inspecting the data or probing into it by identifying elements of
interest, pv的数据选择功能就是为此设计的。

## 6.1了解选择

[selection]{.mark} refers to selecting elements (either cells, points,
rows (in case of tabular datasets), etc.) from datasets. Since data is
ingested into ParaView using readers or sources and transformed using
filters, when you create a selection, you are selecting elements from
the dataset produced as the output of source, filter, or any such
pipeline module.

有很多选择数据的方式：

SpreadSheet View

介绍了一个快速使用数据选择的demo：

Sources-\> Wavelet-\> Render View-\> SpreadSheet View

![](./media/image13.emf){width="5.269360236220472in"
height="3.49837489063867in"}

## 6.2使用视图创建选择

### 6.2.1在RenderView中选择 {#在renderview中选择 .标题3}

有2种方法选择cells, points or blocks：交互式和非交互时

（1）进入非交互式选择模式：![](./media/image14.emf){width="0.28155621172353457in"
height="0.3041240157480315in"}![](./media/image15.emf){width="1.4945538057742782in"
height="0.24091426071741032in"}

（2）交互式选择模式：![](./media/image16.emf){width="0.49770997375328085in"
height="0.2757917760279965in"}

### 6.2.2在SpreadSheet View中选择 {#在spreadsheet-view中选择 .标题3}

### 6.2.3 Selecting in Line Chart View {#selecting-in-line-chart-view .标题3}

## 6.3 Creating selections using the Find Data dialog

## 6.4 Creating selections in Python

## 6.5 Displaying selections

View-\>Selection Display Inspector

## 6.6 Extracting selections

Extract Selection and Plot Selection Over Time

### 6.6.1 Extract selection {#extract-selection-1 .标题3}

![](./media/image17.emf){width="4.666820866141732in"
height="4.018501749781278in"}

### 6.6.2 Plot selection over time {#plot-selection-over-time .标题3}

Plot Selection Over Time

Filter-\>Data
Analysis或![](./media/image18.emf){width="0.2509689413823272in"
height="0.24330489938757655in"}

使用Properties面板的Copy Active Selection之后，修改，然后Apply

Find Data对话框: Only Report Selection Statistics （Plot Selection Over
Time过滤）

![](./media/image19.emf){width="5.313675634295713in"
height="2.6707403762029744in"}

## 6.7 Freezing selections

frustum-based selection

query-based selections created using the Find Data

# 第7章 动画

通过记录一系列的keyframes来创建动画。各frame渲染的几何可保存为pvd文件格式，可返回导入PV，作为时间变化的数据集。

## 7.1动画视图

View-\>Animation View

Animation Keyframes-\>New

## 7.2动画视图header

![](./media/image20.emf){width="5.175482283464567in"
height="0.9075032808398951in"}

几种动画回放模式：

Sequence mode: No. Frames

Real Time mode: Duration

Snap To TimeSteps mode：pv默认的，根据time-varying数据来做动画

可修改动画时钟的数字位数：

Settings/Properties Panel Options/Advanced

## 7.3时间变化数据的动画

使用Animation
View可以将动画时间与数据时间分离，因此可以在动画期间，同时操作数据创建keyframes。

双击TimeKeeper，有3个选择：Animation Time, Constant Time, Variable Time

![](./media/image21.emf){width="2.9693274278215225in"
height="2.3911318897637797in"}

Figure 7.4: Controlling Data Time with keyframes

## 7.4播放动画

设计好动画后，可使用VCR控制条播放动画。

![](./media/image22.emf){width="4.566058617672791in"
height="0.6688199912510936in"}

## 7.5相机动画

[预设置相机的视角轨迹，展示动画。]{.mark}

### 7.5.1插值相机位置 {#插值相机位置 .标题3}

动画不断改变相机位置，2帧之间插值相机的位置。

编辑keyframes，双击track。也可使用Use Current按钮，捕捉当前位置。

![](./media/image23.emf){width="3.6331791338582677in"
height="2.174514435695538in"}

### 7.5.2轨道(Orbit) {#轨道orbit .标题3}

Orbit from the Camera

![](./media/image24.emf){width="2.7303772965879265in"
height="1.9112160979877515in"}

### 7.5.3跟随路径（创建相机跟踪轨道） {#跟随路径创建相机跟踪轨道 .标题3}

![](./media/image25.emf){width="4.661906167979002in"
height="3.747344706911636in"}

[备注：可以考虑使用Xbox等joystick实时控制相机视角，可通过VRPN在VR环境下实施。]{.mark}

# 第8章 保存结果

## 8.1保存数据集

可以保存管线模块产生的数据集，包括：sources, readers, filters.

File-\>Save
Data或者![](./media/image26.emf){width="0.28037292213473314in"
height="0.2718121172353456in"}或者Ctrl+S

或者在pvpython中保存数据：SaveData(\"sample.csv\", source)

## 8.2保存渲染结果

保存截屏：File-\>Save Screenshot

在对话框内设置保存图像的参数。

输出屏幕(Exporting scenes)：File-\>Export View

## 8.3保存动画

File-\>Save Animation

或者(pvpython):

SaveAnimation ('animation.avi', GetActiveView (),

FrameWindow = \[1, 100\] ,

FrameRate = 1)

## 8.4保存[State文件]{.mark}

File-\>Save State...

File-\>Load State...

[有2种State文件：]{.mark}

-   Paraview state file
    (\*.pvsm)：基于xml文档，人和机器可读，PVDM是最可靠的保存应用状态的文件格式。

-   Python state file (\*.py)：可读性强，可手动修改。

加载PVSM文件时，有3个选项：

-   Use File Names From State

-   Search files under specified directory

-   Choose File Names

也可用pvpython加载和保存state文件：

\>\>\> SaveState(\"sample.pvsm\")

\# To load a PVSM state file.

\>\>\> LoadState(\"sample.pvsm\")

还可定义加载状态文件的路径：

\>\>\> LoadState(\"sample.pvsm\", LoadStateDataFileOptions ='Search
files under specified directory ', DataDirectory ='/home/user/sampledata
')

You can use LoadStateDataFileOptions='Choose File Names' too, but you
may need to use the Python trace function in paraview (see Section
1.6.2) to determine the names of the parameters to pass in to LoadState.
They differ among readers.

# 第9章 属性面板

Properties面板是paraview中最常用的面板。

## 9.1属性面板的术语

Properties面板对active对象，即显示active source and active
view的属性，以及显示属性。

### 9.1.1按钮 {#按钮 .标题3}

![](./media/image27.emf){width="3.0765048118985128in"
height="6.258530183727034in"}

### 9.1.2 搜索盒 {#搜索盒 .标题3}

The Search box allows you to search for a property by using the name or
the label for the property. Simply start typing text in the Search box,
and the panel will update to show widgets for the properties matching
the text.

When you start searching for a property by typing text in the Search
box, irrespective of the current mode of the panel (i.e., default or
advanced), all properties that match the search text will be shown.

### 9.1.3 属性 {#属性 .标题3}

## 9.2 自定义布局

Edit-\>Settings

[General]{.mark} tab

# 第10章 颜色图与转换函数

to map the data array to colors, we use a transfer function. A transfer
function can also be used to map the data array to opacity for rendering
translucent surfaces or for volume rendering.

## 10.1基本知识

Color mapping (which often also includes opacity mapping) goes by
various names including scalar mapping and pseudo-coloring. The basic
principle entails mapping data arrays to colors when rendering surface
meshes or volumes.

Since data arrays can have arbitrary values and types, you may want to
define to which color a particular data value maps. This mapping is
defined using what are called color maps or transfer functions.

There are separate transfer functions for color and opacity. Th[e
opacity transfer function is used for volume rendering]{.mark}, and it
is optional when used for surface renderings.

### 10.1.1 Color mapping in paraview {#color-mapping-in-paraview .标题3}

![](./media/image28.emf){width="5.265456036745407in"
height="2.4436964129483814in"}

### 10.1.2 Color mapping in pvpython {#color-mapping-in-pvpython .标题3}

## 10.2 Editing the transfer functions in paraview

View-\>Color Map Editor

![](./media/image29.emf){width="3.019679571303587in"
height="5.37895450568679in"}

Figure 10.2: Color Map Editor panel in paraview showing the major
components of the panel

### 10.2.1 Separate Color Map {#separate-color-map .标题3}

![](./media/image30.emf){width="4.303560804899387in"
height="0.7779024496937883in"}

### 10.2.2 Mapping data {#mapping-data .标题3}

The Mapping Data group of properties controls how the data is mapped to
colors or opacity.

[The transfer function editor]{.mark} widgets are used to control the
transfer function for color and opacity. The panel always shows both the
transfer functions. Whether the opacity transfer function gets used
depends on several things:

-   When doing surface mesh rendering, it will be used only if Enable
    opacity mapping for surfaces is checked

-   When doing volume rendering, the opacity mapping will always be
    used.

### 10.2.3 转换函数编辑器 {#转换函数编辑器 .标题3}

### 10.2.4 Color mapping parameters {#color-mapping-parameters .标题3}

## 10.3 Editing the transfer functions in pvpython

## 10.4 Color legend

Color Map Editor：![](./media/image31.emf){width="0.23680555555555555in"
height="0.30069444444444443in"}

![](./media/image32.emf){width="3.0975in" height="1.6514982502187228in"}

### 10.4.1 Color legend参数 {#color-legend参数 .标题3}

![](./media/image33.emf){width="3.716317804024497in"
height="4.982737314085739in"}

## 10.5 Annotations

## 10.6 Categorical colors

# 第11章 比较可视化

多工况下的比较可视化。

## 11.1 Setting up a comparative view

## 11.2 Setting up a parameter study

# 第12章 可编程过滤

# 第13章 使用Numpy处理数据

编写Python脚本，使用Numpy访问数组并操作。VTK-Numpy整合层可联合使用VTK与Numpy处理数据，尽管与2个系统的数据保证有所不同。

**from** paraview.vtk. numpy_interface **import** dataset_adapter **as**
dsa

**from** paraview.vtk. numpy_interface **import** algorithms **as** algs

# 第14章 远程和并行可视化

## 14.1了解远程处理

假设你有2台电脑：1台在家（配置较低），1台在办公室（配置较好）。可视化数据时：要么将办公室电脑上的数据完全拷贝到在家的电脑，然后按照传统可视化方法可视化数据；一种方法就是：远程处理(remote
processing)。

有2个独立进程：pvserver(运行在办公机器上)，一个paraview客户端（运行在在家电脑上）。他们相互之间通过端口(socket)通信（通过SSH通道）。

所有的数据处理是在pvserver，当需要渲染图形时，paraview可在服务器上做渲染，仅将图像传输到客户端；也可将待渲染的几何体发送到客户端(远程渲染)，做当地渲染(local
rendering)。

## 14.2 paraview中的远程可视化

### 14.2.1启动远程服务器 {#启动远程服务器 .标题3}

在远程系统上启动服务端应用：pvserver

\> pvserver

在终端上可看到设置信息：

Waiting for client \...

Connection URL: cs :// myhost :11111

Accepting connection(s): myhost :11111

这表明服务器已启动，[听]{.mark}从客户端来的连接。

### 14.2.2设置服务器连接 {#设置服务器连接 .标题3}

在客户端打开paraview：File-\>Connect，或者点击![](./media/image34.emf){width="0.2346380139982502in"
height="0.26384186351706035in"}图标，打开Choose Server
Configuration对话框：（刚打开的时候是空的，需要自己[Add Server]{.mark}）

![](./media/image35.emf){width="3.1650798337707786in"
height="2.422349081364829in"}

Figure 14.1: The Choose Server Configuration dialog is used to connect
to a server.

保存的服务器设置是XML文件。可以使用Fetch Servers按钮加载。

### 14.2.3连接到远程服务器 {#连接到远程服务器 .标题3}

选择刚才设置好的服务器，点击Connect，现在可以构建可视化管线了。

### 14.2.4管理多个客户端 {#管理多个客户端 .标题3}

可能同时有多个客户端连接[pvserver]{.mark}。此时，称之为[master]{.mark}，使用管线交互访问。其他客户端仅允许可视化数据。连接的客户端之间由Collaboration
Panel共享信息。

启动pvserver时需要使用参数

pvserver \--multi-clients

如果你是master，可以修改Collaboration Panel中的connect-id:

pvserver \--connect-id=147

master客户端可以停止继续连接：

\--multi-clients \--disable-further-connections

### 14.2.5 Setting up a client/server visualization pipeline {#setting-up-a-clientserver-visualization-pipeline .标题3}

Pipeline Browser

![](./media/image36.emf){width="0.15625in"
height="0.1902176290463692in"}图标旁边的服务器连接地址修改：from
builtin: to cs://myhost:11111

## 14.3 pvpython中的远程可视化

## 14.4 Reverse connections

需要访问防火墙后面的远程计算资源，由外部的客户端访问会有困难。paraview提供reverse
connection

通过File-\>Connect打开Choose Server
Configuration对话框，配置服务器连接时，使用Client/Server (reverse
connection)

第二，启动服务端时：

pvserver -rc \--client-host=mylocalhost \--server-port=11111

输出信息：

Connecting to client (reverse connection requested)\...

Connection URL:csrc://mylocalhost:11111

Client connected.

## 14.5理解并行化处理

### 14.5.1 Ghost levels {#ghost-levels .标题3}

### 14.5.2 Data partitioning {#data-partitioning .标题3}

### 14.5.3 D3 Filter {#d3-filter .标题3}

平衡非结构数据的过滤，创建ghost cells，称之为D3: distributed data
decomposition

Filters-\>Alphabetical-\>D3

## 14.6 Ghost Cells Generator

Ghost Cells Generator过滤，仅产生ghost cells，执行一些算法需要这个。

Build If Required

Use Global Ids

## 14.7ParaView的架构

[Data Server]{.mark} The unit responsible for data reading, filtering,
and writing. All of the pipeline objects seen in the pipeline browser
are contained in the data server. The data server can be parallel.

[Render Server]{.mark} The unit responsible for rendering. The render
server can also be parallel, in which case built-in parallel rendering
is also enabled.

[Client]{.mark} The unit responsible for establishing visualization. The
client controls the object creation, execution, and destruction in the
servers, but does not contain any of the data (thus allowing the servers
to scale without bottlenecking on the client). If there is a GUI, that
is also in the client. The client is always a serial application.

可以以[3种]{.mark}模式运行paraview：

（1）Standalone模式：客户端，dara server, render
server都整合为一个单独的串行程序，都是串行运行。

![](./media/image37.emf){width="1.663565179352581in"
height="1.2650667104111986in"}

（2）Client-Server模式：在并行机器上执行pvserver，在paraview客户端（或pvpython）上连接到服务器。pvserver即有data
server也有render server。

![](./media/image38.emf){width="2.8973370516185475in"
height="1.4320166229221347in"}

（3）Client-render Server-data
Server模式（[很少使用]{.mark}）：所有的3个逻辑单元都以独立程序运行。Client通过单独端口连接到render
server, data server与render server有很多端口连接。

![](./media/image39.emf){width="3.4876509186351705in"
height="1.4947090988626421in"}

## 14.8以paraview和pvpython并行可视化

在paraview或pvpython中使用并行可视化功能，用户必须使用远程可视化，即必须连接到一个pvserver。可使用mpirun启动不止一个进程的pvserver：

mpirun -np 4 pvserver

这样就使用4个进程运行pvserver。它将通过默认端口诊听来自客户端的输入连接。最大的不同是：当数据由source加载后，运行pvserver时，数据将分布到4个进程上（如果数据源是parallel
aware，支持不同进程之间的数据分布）。

然后连接到paraview。使用Source-\>Sphere创建一个球，修改数组为color by
vtkProcessId。将看到一个4种颜色的球体(是parallel
aware)。如果数据源不是parallel aware，使用D3
filter。Source-\>Wavelet，然后Filter-\>Alphabetical-\>D3，点击Apply。然后color
by vtkProcessId，将看到4个颜色分区。

## 14.9使用pvbatch

14.8节已经介绍了并行处理功能，用户必须使用远程可视化，也就是用户必须以Client-Server模式运行ParaView，使用客户端(paraview或pvpython)连接到服务端(pvserver---可使用mpirun并行运行)。但是pvbatch是一个例外。pvpython和pvbatch很相似，都可用python运行脚本程序。与标准的python脚本程序相比，这些可执行程序([指的是pv环境下的python程序]{.mark})可以初始化PV环境，因此用户运行的任何脚本程序都可以[自动定位]{.mark}ParaView
Python模块和库。pvpython与paraview一样，除了没有GUI。你可以认为paraview的GUI被pvpython的Python解译器代替了。另一方面，pvbatch可以认为是pvserver，pvserver执行从[远程客户端]{.mark}的控制命令(paraview或pvpython)。但[在pvbatch中]{.mark}，命令是来自Python脚本，在pvbatch可执行程序中执行。因为pvbatch类似pvserver，不像pvpython，pvbatch可以使用mpirun并行运行。在这种情况下，根进程（rank==0）起到客户端的作用，解译python脚本执行相关命令。因为pvbatch设计成起到自己服务端的作用，用户不能以python脚本形式连接到远程服务端，也就是不能使用simple.Connect。并且，pvbatch设计为批处理操作，意思是用户仅定义Python脚本作为命令输入参数。不像pvpython，用户不能以交互的shell方式输入Python命令来运行程序。

\# process the sample.py script in single process mode.

\> pvbatch sample.py

\# process the sample.py script in parallel.

\> mpirun -np 4 sample.py

通常，使用解译器交互式输入命令时使用pvpython，如果需要并行化运行时使用pvbatch。

[备注]{.mark}：RegESM中通过ESMF将各进程上的模式计算数据输入到ParaView中并行化渲染，然后后处到客户端，通过Catalyst在线可视化。

## 14.10获取数据到客户端

3.3节介绍了如何获取数据对象的信息，但没有介绍如何访问数据对象本身。

可以使用Python脚本访问数据对象。python脚本以pvpython或paraview在客户端运行，将数据从服务端移动到客户端：

**from** paraview.simple **import** \*

Connect(\"myhost\")

\# Create a sphere source on myhost

s = Sphere ()

full_sphere = servermanager .Fetch(s)

## 14.11渲染

现在的中等显卡已支持快速的3D图形渲染，但数据量越大，渲染速度越慢。

为保证实时交互式渲染，ParaView支持2种渲染模式，需要时可自动切换。第1种是[静态渲染]{.mark}，数据以高的Level
of Detail
(LOD)渲染，这种模式保证所有数据都精确地标准。第2种是[交互式渲染]{.mark}，相对精度，优先考虑渲染速度，这种模式努力提供快速的渲染速度而忽视数据大小。

当用户使用3D视角（例如：使用鼠标做旋转，平面或放大），ParaView就是使用交互式渲染。这是因为，在交互视图时，高的帧速率是必须的，使图形特征可识别，因为每一帧立即被新的渲染所代替，在这种模式下细节是不重要的。当3D
View交互式渲染不发生的情况，ParaView使用静态渲染。当使用鼠标拖动3D
View时，看到的是近似渲染，当释放鼠标后，全部细节将呈现。

交互式渲染是[速度和精度]{.mark}之间的折中。当使用较低的LOD时需要关注很多[渲染参数]{.mark}。

### 14.11.1基本渲染参数 {#基本渲染参数 .标题3}

Level of Detail
(LOD)渲染参数是最重要的。当使用交互式渲染时，几何体被较低的LOD代替，使用较少多边形的近似几何体。

3D渲染参数设置：[Edit-\>Setting]{.mark}，对话框中的渲染选项位于[Render
View]{.mark}（如下图），包括：LOD Threshold、LOD
Resolution、Non-Interactive Render Delay、Use Outline For LOD
Rendering。

![](./media/image40.emf){width="2.9528772965879266in"
height="3.6581288276465442in"}

ParaView还有更多的渲染参数。这里仅列举一些影响渲染效率，不管ParaView是否是以client-server模式运行，有：

-   Translucent Rendering Options (Depth Peeling, Depth Peeling for
    > Volumes, Maximum Number Of Peels)

-   Miscellaneous (Outline Threshold, Show Annotation)

以上设置都在![](./media/image41.emf){width="0.2693132108486439in"
height="0.24917213473315836in"}按钮中。注意：上面的渲染参数没有列全，没有列出显著影响渲染效率的参数，也没有列出并行化client-server渲染的设置参数，将在14.11.4节讨论。

### 14.11.2基本的并行化渲染 {#基本的并行化渲染 .标题3}

ParaView使用并行化渲染库IceT。IceT使用sort-last算法做并行化渲染，该种渲染算法是[在各进程上独立渲染其几何部分，然后合并分部图像，形成最终图像。]{.mark}

![](./media/image42.emf){width="3.969025590551181in"
height="1.9534623797025372in"}

IceT还包含多种其他的并行化图像合成算法，比如：binary tree, binary swap,
radix-k，使用多个阶段(phases)将工作分到各进程。

sort-last算法的优势是其渲染效率对考虑的数据规模完全不敏感，这使得该算法非常的可扩展，适合于大数据渲染，但是并行渲染[开销(overhead)]{.mark}随图像的像素数线性增大，因此需要设置一些渲染参数来处理图像大小。

IceT还具有驱动[tiled
display]{.mark}的功能，是由一系列显示器和投影仪分片组合成高分辨率的大型显示。使用sort-last算法做分片显示有点反直觉，因为合成的像素数很大。但是，IceT设计成利用各进程上的数据空间位置，能显著减小合成的量。空间位置可以使用Filters-\>Alphabetical-\>D3过滤功能对数据强制实施。

可以关闭并行渲染功能，但这仅对小规模数据渲染的情况。

### 14.11.3 Image Level of Detail {#image-level-of-detail .标题3}

并行化渲染的交互期间，ParaView可subsample图像，降低通信量。

![](./media/image43.emf){width="4.848958880139983in"
height="1.2208136482939633in"}

Finest LoD 2 4 8

从左到右，分辨率依次降低因子：2, 4, 8

另外，图像从server传输到client之前还可以压缩，paraview使用4种压缩方法，降低传输的数据量，最大利用集群带宽：

-   LZ4压缩：高速压缩和解压缩算法。

-   Squirt: Sequential Unified Image Run Transfer，降低color
    > depth，增加run length。

-   Zlib:实施Lempel-Ziv算法，压缩想过比Squirt好，但压缩时间更长，增加了延迟。

-   NVPipe: NVIDIA硬件加速压缩和解压缩的库，需要Kepler级以上的NVIDIA
    > GPU硬件。

### 14.11.4并行化渲染参数设置 {#并行化渲染参数设置 .标题3}

在Render
View标签中，与其他几个渲染选项混在一块。并行化渲染的参数选项有：

-   Remote/Parallel Rendering Options

-   Client/Server Rendering Options

-   Image Compression

![](./media/image44.emf){width="3.463963254593176in"
height="4.167272528433946in"}

### 14.11.5大规模数据渲染参数设置 {#大规模数据渲染参数设置 .标题3}

默认的渲染参数适合于大多数用户。但当处理非常大规模数据时，需要设置渲染参数。参考手册214页。

# 第15章 内存侦测

# 第16章 多块侦测

# 第17章 注释(annotations)

Explicit labeling and annotation of particular data values is often an
important element in data visualization and analysis. ParaView provides
a variety of mechanisms to enable annotation in renderings ranging from
free floating text rendered alongside other visual elements in the
render view to data values associated with particular points or cells.

## 17.1注释源

Sources-\>Alphabetical

![](./media/image45.emf){width="4.066247812773403in"
height="0.897179571303587in"}

## 17.2注释过滤

# 第18章 坐标轴网格

To turn on the [Axes Grid]{.mark} for a Render View, you use the
[Properties]{.mark} panel. Under the View section, you check the Axes
Grid checkbox to turn the Axes Grid on for the active view.

![](./media/image46.emf){width="3.0789304461942257in"
height="0.6883967629046369in"}

# 第19章 用户个性化ParaView

## 19.1设置

Edit-\>Settings：有很多tab：

General

![](./media/image47.emf){width="3.8995844269466318in"
height="5.835652887139108in"}

Camera设置：Rotate, Pan, Zoom, etc.

![](./media/image48.emf){width="3.829031058617673in"
height="4.313934820647419in"}

Render View设置：参考14.11.1和14.11.4节

Color Palette

![](./media/image49.emf){width="3.9454615048118984in"
height="4.349886264216972in"}

![](./media/image50.emf){width="2.855916447944007in"
height="3.475644138232721in"}

Figure 19.5: Popup menu allows you to link a color property to a color
palette category in the Properties panel.

## 19.2用户的默认设置

用户定义的默认属性：Properties, Display, and View

定义背景颜色：View-\>Properties

使用JSON配置默认设置：˜/.config/ParaView/ParaView-UserSettings.json

![](./media/image51.emf){width="2.8217989938757655in"
height="2.239549431321085in"}

Figure 19.6: Buttons for saving and restoring default property values in
the Properties panel
