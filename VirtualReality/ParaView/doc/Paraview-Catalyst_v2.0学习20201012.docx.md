![](./media/image1.emf){width="4.107166447944007in"
height="1.8695417760279964in"}

# 1 前言

近年来，计算系统在速度和能力方面快速增长。大规模计算在成千上万个处理器的并行系统上执行。因此，并行计算方法和系统是现代计算模拟的核心。随着计算核心数的增加，输入输出（IO）成为限制因子。计算能力和IO带宽对大规模计算产生深刻影响。

传统的模拟过程分为：前处理（划分网格、设置边界条件和计算参数等）、计算和后处理（分析和可视化结果）。3个过程独立，但读写大数据称为大型计算的瓶颈。

![](./media/image2.emf){width="4.374370078740157in"
height="2.8466557305336835in"}

图1.1在桌面电脑上启用6进程在某些分析计算和保存完整数据耗时的比较

当以较低频率保存结果，例如非恒定分析，每10步保存1次结果，将放弃90%的数据。有时，甚至保存单步的完整数据也将超出IO系统的能力。

Catalyst方法将传统的3步整合为1步，即将后处理直接整合到计算过程中：

![](./media/image3.emf){width="5.338859361329834in"
height="1.9376159230096237in"}

图1.2 传统的(a)和Paraview Catalyst (b)的工作流程

不用保存完整的数据集到磁盘，IO只要提取相关的信息变量即可。数据提取，诸如等值线（面）、切面或流线等，比完整数据集要小几个数量级。因此，输出提取数据将显著降低IO成本。

![](./media/image4.emf){width="4.351054243219598in"
height="2.293232720909886in"}

图1.4 比较保存完整数据集与保存特殊分析输出的文件大小（单位：bytes）

Co-processing还可以及时发现问题，停止计算。

使用Catalyst做[in-situ可视化]{.mark}的典型流程是：使用Paraview
GUI定义模拟的输出，创建Python脚本实施初始化；然后，当模拟启动时，加载该脚本；然后，在模拟执行期间，以同步方式（即模拟在运行中）生成分析和可视化输出。

Catalyst可生成图片，计算统计量，生成制图和提取衍生信息，如多边形数据或等值面，来可视化几何形体或数据。

具体实施Catalyst时考虑2方面：（1）是够降低IO成本是重要的？（2）是够这些管线合理地尺度化了？否则可视化时会使模拟过程阻塞，危害或影响整体分析周期时间。但是，Paraview和VTK系统都是并行化的，通常是用于大部分的应用。

# 2 用户使用Catalyst

![](./media/image5.emf){width="4.873167104111986in"
height="1.6169838145231845in"}

图2.1传统的工作流程（蓝色）和Catalyst增强的工作流程（绿色）

使用Catalyst增强的工作流程，用户预设需要的可视化和分析输出，作为一个预处理步。在模拟运行时生成输出，之后用于用户的分析。Catalyst输出可以有多种格式，诸如使用伪颜色制作的几何形体的渲染图形、绘制图形（如条状图、线状图等）、数据提取（如等值面、切面或流线等）以及计算量（如最大剪切力、升力等）。

用户有2种方式使用Catalyst实施[in-situ可视化]{.mark}：第1种，定义一套参数，传入预设的Catalyst管线；第2种，用户在Paraview
GUI中创建Catalyst管线脚本。

## ParaView的在线可视化实现

ParaView使用VTK来处理数据和作为渲染引擎，使用Qt开发用户界面。使用Python语言实现完全的脚本操作。通过Python对引擎做的所有改变都自动反馈到用户界面。

ParaView[以batch作业]{.mark}来运行。可以通过编写接口的XML描述，或者编写C++类增加其他的模块。XML接口允许用户或开发者增加自己的VTK过滤器，且无需编写特别的代码和重新编译。

ParaView以MPI并行运行，包括集群、可视化系统、大型服务器和超算。ParaView使用数据并行模型，其中数据分解为pieces，由不同进程处理。使用ghost
points/cells通信，交换相邻信息。还支持[分布式渲染]{.mark}（数据在各计算节点上渲染，之后使用[depth
buffer]{.mark}合成）、当地渲染（在一个计算节点上收集计算得到的多边形，当地渲染）以及两者混合使用（例如level-of-detail模型，可以当地渲染，而完整模型以分布式方式渲染）。

用户界面（ParaView-GUI）使用client/server模式在独立的电脑上运行。以这种方式，用户可以利用远程的HPC渲染集群。ParaView
Client是串行应用程序，总是以paraview命令运行。Server是MPI并行程序，必须用并行化作业来启动。有[2种模式]{.mark}来启动ParaView
Server：

（1）所有数据处理和渲染都以相同的并行作业运行，使用pvserver命令启动server

（2）数据处理是以1个并行作业处理，而渲染是以另一个并行作业处理，分别使用pvdataserver和pvrenderserver命令。将数据处理与渲染分开的目的是：可以使用2个不同的并行计算机，一个具有高端的CPU，一个具有高端的GPU。但是将Server功能分成2个部分需要重新分区数据和两者之间转移数据，这造成的开销，相对以相同作业下实施数据处理与渲染，一般不可忽略。因此，建议几乎所有的应用就简单地用一个server执行。

ParaView还可作为web应用程序使用，目的是为3D可视化提供[远程协作]{.mark}的web接口(ParaViewWeb)。

## 预设Catalyst管线

这部分工作主要由模拟开发者来完成，将方便模拟使用者，降低使用门槛。概念就是：对于大多数过滤器（[filters]{.mark}），仅需要设置有限数目的参数即可。例如，对于切面过滤，用户仅需要设置穿过平面的点和平面的法向。另一个例子是，设置阈值过滤器，其中仅需要定义变量和范围。对每一个管线，参数还需要包含[待输出的文件名和输出频率]{.mark}。

## 创建Catalyst管线Python脚本

> 这种方法允许用户可自由添加增多的功能来控制输出。用户可在Paraview
> GUI中创建他们自己的Catalyst Python脚本。

这种方式需要2个条件：

（1）Paraview启用了[CoProcessing Script
Generator插件]{.mark}（编译时默认启用），注意：插件版本应对应Catalyst版本。

（2）用户已有代表性数据集，由此开始：读入的数据集有相同的数据类型（vtkUnstructuredGrid,
vtkImageData等），与模拟[适配器(adaptor)]{.mark}代码中网格上定义的属性相同。

[在GUI中创建Catalyst Python脚本的步骤]{.mark}如下：

（1）加载创建脚本的插件:Tools-\>Manage Plugins...-\>
CatalystScriptGeneratorPlugin -\> Writes and CoProcessing

（2）加载代表性数据和创建脚本：不是真的要输出文件，而是创建将在模型运行期间将要在哪输出文件。在Writers菜单下，选择提取数据的合适信息。用户应定义文件名及输出频率。文件名必须包含%t，当创建文件时，将被时间步代替。![](./media/image6.emf){width="0.45147200349956257in"
height="0.19826443569553806in"}

![](./media/image7.emf){width="3.4026935695538056in"
height="3.56791447944007in"}

图2.2
一个管线有2个writer的例子：第1个从Calculator过滤输出，第2个从切面过滤输出

（3）当创建了完整的Catalyst管线后，必须[从Paraview输出Python脚本]{.mark}。

选择CoProcessing菜单下面的Export
State控件。用户可点击跳出的初始化窗口中的Next按钮。![](./media/image8.emf){width="1.698956692913386in"
height="0.2362817147856518in"}

（4）完成以上步骤，用户必须选择源（即没有输入的管线对象），适配器将创建和添加这些源到输出。注意：通常这不包括从[Sources]{.mark}菜单，因为生成的Python脚本将实例化这些对象（例如画流线的seed）。图2.2中的源是filename_10.pvti读取器，类比于模型代码适配器提供的输入。用户或者点击左box中的源，添加到右box，或者选中左box中源，点击[Add]{.mark}。如图2.3。选中所有的源后，点击[Next]{.mark}。

本例中，选择filename_10.pvti源，做法：

![](./media/image9.emf){width="4.081903980752406in"
height="2.4612128171478567in"}

图2.3 选择filename_10.pvti作为Catalyst管线的输入

（5）标记输入。点击Next

![](./media/image10.emf){width="4.096176727909011in"
height="2.4698173665791776in"}

图2.4 提供Catalyst输入的标识字符串

（6）允许Catalysy检查Live Visualization连接，从不同视图输出屏幕图像。

Paraview Live Visualization和Cinema在后文讨论。

对于[屏幕显示(Screenshots)]{.mark}，有多个选项，见图2.5：

第1个全局选项是Rescale the lookup
table，将伪颜色尺度化到数据范围。每个视图的其他选项有：

Image Type --- 输出到屏幕的图片格式；

File Name -- 创建的文件名，必须包含%t，实际模拟时间步将代替%t；

Write Frequency -- 创建屏幕显示的频率；

Magnification -- 用户将创建比当前Paraview GUI分辨率更高的图像；

Fit to Screen --
指定是否将数据扩展到合适的屏幕显示，类似于GUI中点击[扩展]{.mark}按钮。

![](./media/image11.emf){width="4.50755249343832in"
height="3.6094346019247596in"}

图2.5 设置输出屏幕图像的参数

如果有多个视图，用户需逐个拖入，使用Next View和Previous
View按钮。完成后，点击Finish按钮，创建Python脚本。

（7）最后1步：指定创建的Python脚本文件名。

### 创建代表性数据集 {#创建代表性数据集 .标题3}

有2种方法：

-   第1种，运行带输出完整网格及完整属性信息的Catalyst的模拟；

-   第2种，在Paraview中使用源和过滤器。

### 操纵Python脚本 {#操纵python脚本 .标题3}

需要会使用Python编程的开发者。

## Paraview Live

除了能提前设置管线，通过Paraview的Live功能，分析者可通过Paraview
GUI连接到模拟，可修改当前管线。这对于通过修改管线，改善Catalyst-模拟输出信息的质量。使用![](./media/image12.emf){width="1.2816032370953632in"
height="0.23536417322834646in"}完成连接，这将连接模拟到pvserver。完成连接后，GUI管线如图2.6。

![](./media/image13.emf){width="3.501330927384077in"
height="1.9446609798775154in"}

图2.6 现场连接的Paraview GUI管线

![](./media/image14.emf){width="0.2863046806649169in"
height="0.176078302712161in"}：表示没有Catalyst提取数据发送到服务器；如果需要发送，点击此按钮；![](./media/image15.emf){width="0.24027777777777778in"
height="0.20833333333333334in"}：表示已经在Paraview服务器上可获取数据了。停止提取发送给服务器的信息，可以在Paraview服务器管线中删除该对象。

例如，Contour0 --- 已经发送给pvserver

允许模拟过程中，暂停Catalyst功能的模拟。

## Cinema-V4.2引入

Cinema是一种[基于图像(image-based)]{.mark}的方法来在线分析和可视化的。概念就是：将一个组织好的图像集保存在Cinema数据库中，分析者可从生成的图像中实施事后分析和可视化。可使用Catalyst创建Cinema数据库，可通过Paraview
GUI的[Catalyst Script
Generator插件]{.mark}来定义Cinema输出。图2.7显示了启用Cinema输出的扩展选项。选项有：

Export Type \-\-\--
该选项定义当生成图像时，应如何操纵视图相机。当前选项包括：[None, Static,
Spherical]{.mark}。None表示对该视图不需要输出Cinema；Static表示不移动相机；Spherical将围绕视图中心，以一定的Phi和Theta角度旋转相机。

Cinema Track Selection \-\--
该选项允许改变过滤器参数，以及用于伪颜色生成的场数据。通过旋转左栏中的管线对象，users
can specify the arrays to pseudo-color by in the right pane's Arrays tab
or the filter's parameters in the right pane's Filter Values tab.
注意，当前仅使用[Slice, Contour和Cut过滤器]{.mark}修改过滤器值。

![](./media/image16.emf){width="2.882155511811024in"
height="4.049837051618548in"}

图2.7 Catalyst输出选项的Cinema输出

## 避免数据爆炸

创建管线时，过滤器的选择和量级会[极大地影响Catalyst]{.mark}和Paraview的计算效率。当发送大规模数据到Paraview服务器时，由于内存限制，由于内存不足会导致计算终止。[最坏的情况]{.mark}就是：发送一般化的网格数据结构，如非结构网格；而结构网格数据很紧凑。下面罗列出常用过滤器的内存使用效率分类：

1、几乎不占用多少内存的：

![](./media/image17.emf){width="5.437514216972878in"
height="1.0291240157480315in"}

2、添加场数据---使用相同的网格，但需要存储额外的变量。

![](./media/image18.emf){width="5.502648731408574in"
height="0.8947233158355206in"}

![](./media/image19.emf){width="5.552648731408574in"
height="0.8402657480314961in"}

3、拓扑关系改变，降维\-\--输出多边形数据，但输出单元是1个或多维，但少于输入单元维度。

![](./media/image20.emf){width="5.6132185039370075in"
height="0.8337554680664917in"}

4、拓扑关系改变，中等程度降维\-\--降低数据集的单元总数，但输出多边形或者非结构网格格式。

![](./media/image21.emf){width="5.650840988626421in"
height="0.5294444444444445in"}

5、拓扑关系改变，没有降维\-\--
当改变数据集的拓扑关系，不减小数据集的单元数，输出多边形或非结构网格格式。

![](./media/image22.emf){width="5.564016841644794in"
height="1.4290660542432196in"}

当创建管线时，过滤器通常以某种方式组织，来限制数据爆炸。例如，管线应先组织，以降低维度。另外，[降低维度]{.mark}比提取数据更好（例如，Slice过滤器比Clip过滤器更倾向使用）。仅当降低一个或多个数量级的数据规模时才使用提取(extract)。当输出提取的数据时，能使用[subsampling]{.mark}（例如，Extract
Subset过滤器或Decimate过滤器），来降低文件大小，但需注意降低数据规模不应隐藏细节特征。

# 3 开发者的Catalyst

![](./media/image23.emf){width="4.45462489063867in"
height="0.5438670166229221in"}

编写适配器代码的开发者应该了解[模型代码的数据结构、VTK数据结构和Catalyst
API]{.mark}。

## High-Level View

连接Catalyst与模型代码需要大量的编程，对代码的影响很小。大多数情况，仅需编写[3个函数]{.mark}，在模型代码中调用：

1.  Initialize

2.  CoProcess

3.  Finalize

示例代码：

![](./media/image24.emf){width="5.331128608923884in"
height="2.7093110236220475in"}

适配器代码在单另文件中实施。适配器代码负责模型代码与Catalyst之间的接口。除了初始化和结束Catalyst外，适配器代码的另一作用是：

（1）查询Catalyst，看是否需要实施co-processing；

（2）提供调整用于co-processing的[网格和场]{.mark}的VTK数据对象。

下面的伪代码显示了适配器的大致内容：

![](./media/image25.emf){width="5.311836176727909in"
height="2.142363298337708in"}

下面的示例代码显示了简化版的适配器。后面更详细的讨论API的细节：

![](./media/image26.emf){width="4.571499343832021in"
height="4.0849726596675415in"}

![](./media/image27.emf){width="5.434853455818023in"
height="7.61174978127734in"}

在了解VTK和Catalyst
API（在编写适配器代码时需要）的细节之前，应该了解一些细节信息：

（1）VTK编号从0开始；

（2）vtkIdType是在Catalyst配置时设置的一个整型类型；

（3）VTK手册；

（4）Paraview使用手册。

VTK是一种普适性工具，有很多种表征网格的方法。VTK需要普适性的原因就是：能够处理拓扑复杂的非结构网格，也能处理简单网格（如均匀的笛卡尔网格等）。以下是VTK支持的网格类型：

![](./media/image28.emf){width="4.901194225721785in"
height="3.153050087489064in"}

图3.1 VTK数据集类型

此外，VTK还支持很多2D/3D单元类型，如三角形、四面性、四面体、金字塔、棱柱、六面体等。VTK还支持数据集中各节点或单元上的相关场信息。VTK中，一般称为属性数据(attribute
data)，当与数据集中的节点或单元相关时，称之为点数据(point
data)或单元数据(cell data)。

VTK数据集的整体结构包含：网格信息，与网格上各节点和单元相关信息的数组。如下图：

![](./media/image29.emf){width="4.171585739282589in"
height="2.2027602799650046in"}

图3.2 VTK数据集

## VTK数据对象API

VTK使用管线架构来处理数据。编写适配器代码需要了解管线架构。

### vtkObject {#vtkobject .标题3}

几乎所有的VTK类都是由vtkObject派生的。

### vtkDataArray {#vtkdataarray .标题3}

concrete implementation ( )

### GridTypes {#gridtypes .标题3}

vtkPolyData, vtkUnstructuredGrid, vtkStructuredGrid, vtkRectilinearGrid
and vtkImageData/vtkUniformGrid

![](./media/image30.emf){width="5.6396544181977255in"
height="1.930340113735783in"}

图3.4 VTK数据集的类分级

## VTK管线

关于VTK管线架构的完整介绍参考VTK的用户手册。本质上，创建从Catalyst的输出就是创建VTK管线，在模型运行期间，在某点上执行VTK管线。VTK使用数据流的方法将信息转换为需要的形式。需要的形式可能是：衍生计算量、提取变量或者图形信息。VTK中的过滤器实施转换。这些过滤器(filter)读取数据，基于输入过滤器的一套参数执行操作。大多数过滤器执行具体的操作，但将多个过滤器串接起来，可完成很多种转换数据的操作。

只有来自其他过滤器的输入的过滤器，称为[源]{.mark}；不发送任何输出给其他过滤器的过滤器，称为[汇]{.mark}。[源过滤器]{.mark}可能是一个文件读取程序，[汇过滤器]{.mark}可能是一个文件写出程序。称这样的一套连接的过滤器为[管线]{.mark}。对于Catalyst，[适配器起到所有管线的源过滤器的作用]{.mark}。如下图所示：

![](./media/image31.emf){width="5.245791776027996in"
height="1.5796062992125983in"}

图3.5 过滤器链条

管线的作用就是在过滤器之间传递[vtkDataObject]{.mark}。管线可视为有方向的、不循环的图。VTK的管线的一些重要特性如下：

（1）不允许过滤器修改他们的输入数据对象；

（2）当下游过滤器要求他们执行的时候，该过滤器才执行；

（3）如果请求改变或上游过滤器变化，该过滤器才重新执行；

（4）过滤器可以有多个输入和输出；

（5）过滤器可以向多个单独的过滤器发送他们的输出。

这将从几个方面影响Catalyst。第一，当构建VTK数据结构时，适配器可使用当前内存；第二，当具体要求管线执行时，该管线才重新执行。

## Catalyst API

下面介绍适配器如何将信息在模型代码与Catalyst之间传递信息。信息可分解为3个部分：

1.  VTK数据对象；

2.  管线；

3.  控制信息。

### High-Level View {#high-level-view-1 .标题3}

第一步是[初始化，设置Catalyst环境，创建之后要执行的管线]{.mark}。这一般在执行完MPI_init(
)之后立即执行。

第二步是[执行管线]{.mark}（如果需要）。这步一般在各时间步更新后执行。

最后一步是[结束Catalyst]{.mark}，一般在执行MPI_Finalize( )之前执行。

第一步和最后一步很简单，但中间一步有点复杂。原则上，中间步要查询，看看是否要求执行。如果不要执行，则立即返回控制到模型代码。如果一个或多个管线需要重复执行，则适配器需要更新表征网格和属性信息的VTK数据对象，然后执行要求的管线。根据管线中过滤器需要完成的工作量，这要花费的时间长短不一。当所有需要执行的管线完成后，控制返回到模型代码。

### Class API {#class-api .标题3}

vtkCPProcessor

vtkCPPipeline

## 适配器：把所有的放一起

下面介绍在适配器中如何将上述的类整合在一起。

1、初始化步骤：

（1）创建vtkCPProcessor对象，调用Initialize( )；

（2）创建vtkCPPipeline对象，将他们添加到vtkCPProcessor。

2、调用co-processing子程序：

（1）创建vtkCPDataDescription对象(\*)；

1)调用SetTimeData();

2)对每一个输入数据对象，调用AddInput()，使用关键标识字符串(\*)；

3)选择性地，调用SetForceOutput()。

（2）调用vtkCPProcessor::RequestDataDescription()，使用创建的vtkCPDataDescription对象；

1)如果RequestDataDescription()返回0，则返回控制到模型代码；

2)如果RequestDataDescription()返回1，则：

2.1：对每一个vtkCPInputDataDescription创建vtkDataObject和属性，添加他们，使用vtkCPDataDescription::GetInputDataDescriptionByName(const
char\* name)-\>SetGrid()

2.2：调用vtkCPProcessor::CoProcess()

（3）结束步：调用vtkCPProcessor::Finalize()，并删除vtkCPProcessor对象。

注意：[带星号(\*)的项目]{.mark}，在首次执行co-processing子程序后完成，之后保持不变的数据结构。下面通过一个示例代码说明一个简单的适配器程序：

![](./media/image32.emf){width="3.8991666666666664in"
height="6.488000874890639in"}

![](./media/image33.emf){width="4.770984251968504in"
height="5.768564085739283in"}

## 与C或FORTRAN的模型代码连接

Catalyst主要是包含很多类的C++库，用Python做了封装。这使得Catalyst连接C++和Python的模拟代码就很容易。很多模型代码是C或FORTRAN开发的，需要增加一些C++代码，来创建VTK数据对象。

在C++函数声明前，增加[extern
"C"]{.mark}，被FORTRAN或C代码调用。对于头文件，按照以下编写：

![](./media/image34.emf){width="5.096645888013998in"
height="1.594303368328959in"}

对于FORTRAN代码，所有的数据对象都以指针传递。对于C代码，如果不用Python，添加合适的头文件CAdaptorAPI.h，主要定义的函数有：

![](./media/image35.emf){width="5.341053149606299in"
height="1.716836176727909in"}

## 编译和链接源代码

最后一步就是：编译所有的部件，并将他们链接起来。

最简单的编译方法就是使用[CMAKE]{.mark}，编写CMakeLists.txt，示例如下：

![](./media/image36.emf){width="4.895950349956255in"
height="4.2358169291338585in"}

使用Catalyst的参数：USE_CATALYST，编译Paraview。此时，将在FEDriver模拟代码示例，安置对Adaptor的依赖库。

如果模拟代码不需要Python接口到Catalyst，用户可避免Python依赖，做法是：改变Paraview组件，[从vtkPVPythonCatalyst变为vtkPVCatalyst]{.mark}。这些组件也会引入
其他的Catalyst组件和头文件，用于编译和链接。

如果不用CMAKE编译，那就要找一个示例，确定需要的用于编译的头文件路径和需要链接的库。

# 参考文献

[Data Co-Processing for Extreme Scale Analysis Level II ASC
Milestone]{.mark}. David Rogers, Kenneth Moreland, Ron Oldfield, and
Nathan Fabian. Tech Report SAND 2013-1122, Sandia National Laboratories,
March 2013.

[The ParaView Coprocessing Library: A Scalable, General Purpose *In
Situ* Visualization Library]{.mark} Nathan Fabian, Kenneth Moreland,
David Thompson, Andrew C. Bauer, Pat Marion, Berk Geveci, Michel
Rasquin, and Kenneth E. Jansen. In *IEEE Symposium on Large-Scale Data
Analysis and Visualization (LDAV)*, October 2011, pp. 89-96.

# CatalystExampleCode学习记录

一般由几个文件构成：

-   模拟程序驱动：FEdriver.c，包括其他的一些子程序或.h文件

-   [适配器程序：FEAdaptor.cxx]{.mark}
    （只能用C++编程，因为ParaView的API都是C++编程的）

-   CMakeLists.txtx文件

## CFullExample

使用了Catalyst的一些方法来存储VTK数据结构。

假设为vtkUnstructuredGrid

[FEDriver.c]{.mark}

#ifdef USE_CATALYST

CatalystInitialize(argc, argv);

#endif

## CFullExample2

改进了CFullExample，显式存储VTK数据结构。

假设为vtkUnstructuredGrid

[FEDriver.c]{.mark}

## Fortran90FullExample

在一个FORTRAN语言的模拟程序中连接Catalyst
