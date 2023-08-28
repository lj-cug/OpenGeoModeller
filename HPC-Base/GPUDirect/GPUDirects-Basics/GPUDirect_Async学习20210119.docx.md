# GPUDirect Async技术

## 摘要

GPUDirect是Nvidia公司开发的[一系列]{.mark}的优化多GPU间（P2P）或多GPU与第三方设备（RDMA）之间[数据移动]{.mark}的技术。

[CUDA 8.0]{.mark}开始引入GPUDirect
Async技术：允许GPU与第三方设备之间直接的同步(synchronization)。

Async允许直接trigger and poll排队进入Infiniband
Connect-IB网卡的通信操作来完成。

## 1前言

CUDA
3.1在2010年，引入GPUDirect技术，通过[共享的页锁定内存]{.mark}，加速第三方PCIe网络和存储设备间的通信。

CUDA 4.0在2011年，引入GPUDirect [Peer-to-peer
(P2P)技术]{.mark}，允许在同一个PCIe root
port下多GPU间的直接访问和数据转移。几乎同时，[CUDA-aware
MPI]{.mark}增加更多支持P2P，加速[多节点间GPU间]{.mark}的通信。

CUDA 5.0引入[GPUDirect
RDMA]{.mark}，启用多GPU（Kepler架构）与第三方外围设备（如Network
Interface Controller, NIC）之间直接的PCIe数据转移。

MLNX_OFED 2.1，Mellanox在ConnectX-3和Host Channel Adapter
(HCA)上支持GPUDirect。

在2015年超算大会上，发布[GPUDirect Async]{.mark}技术，相应地在CUDA
8.0中引入了需要的API。

从MOFED 3.4, Mellanox发布了一套[Peer-direct Async
Verbs]{.mark}扩展，作为对GPUDirect Async的补充。

2019年，发布了[GPUDirect
Storage]{.mark}技术，实现GPU到Nvme固态硬盘的直接数据存储。

传统的多GPU加速计算应用中，在完成CUDA核函数计算后，在发送相关通信到NIC上之前，执行cudaStreamSynchronize()；或者等待CPU通信完成，在执行GPU计算任务之前，执行MPI_Wait()。

GPUDirect P2P和RDMA技术都是优化GPU与NIC之间数据转移。

GPUDirect
Async技术不是卸载CPU的数据转移路径到GPU上，[而是使GPU]{.mark}直接通过PCIe总线启动通信转移和同步，不使用运行在CPU上的通信代理(agent)，这就潜在地改善了计算效率（允许更多地重叠计算和通信），释放CPU核心循环计算（可执行其他任务），或者改善了使用低配置CPU的异构并行计算效率。

GPUDirect Async技术的2种不同方式：

（1）Stream Asynchronous (SA)

引入on-a-stream的点对点(point-to-point)和单边(one-sided)通信原语，融合通信和计算，以统一的CUDA任务同步方式执行，属于CUDA流的概念范畴。换句话说就是，以提交到CUDA流上的顺序执行通信，正确地与CUDA异步内存复制和CUDA核函数执行混合在一起。分解为以下4个阶段：

A、CPU准备通信原语，包括buffer pointers、size、network
address等，以及post相关的描述(descriptors)到NIC命令队列；

B、收集用于激活这些描述的元数据(meta-data)，转换为CUDA内存操作(MemOps)；

C、CPU将这些MemOps提交给用户CUDA流；

D、稍后，CUDA流执行这些MemOps，启动在第a步准备好的通信。

（2）Kernel-Initiated (KI)

KI是对SA的改变，其中第A步和第B步与SA一样。

C、CPU将元数据传给CUDA核函数；

D、稍后，经过一些形式的inter-thread同步（具体针对某种计算），CUDA核函数（与CUDA流相反）使用这些元数据，或者启动通信，或者等待通信完成。

注意，在SA和KI两种方式中，第A步都是在CPU上执行，使用相同的程序库(software
stack)处理正常的通信，例如Mellanox user-space
[libmlx5]{.mark}驱动程序，代码都是逐位(bitwise)操作和分支(branch)，与NIC硬件有关（每种NIC厂商都有自己的HW接口），难以并行化，因此在CPU上方便执行优化，具有低延迟。

## 2相关研究

![](./media/image1.emf)

Ohio State University团队在MVAPICH2 (MPI-GDS)中引入GPUDirect
Async，特点是在探讨GPUDirect Async功能的同时，不修改MPI
API。本文探讨了one-sided通信原语和CUDA核函数初始化的通信，并做出了改进。

## 3 GPUDirect Async

如图1，显示了使用GPUDirect RDMA-enabled的Infiniband HCA：

![](./media/image2.emf)

![](./media/image3.emf)

图1 GPUDirect RDMA的计算和发送流程图

注意，如果没有使用GPUDirect
RDMA，在通信任务中就要使用设备到主机的数据拷贝(cudaMemcpyDevicetoHost)。

GPUDirect Async通过使用GPU启动(1) 与HCA的通信以及(2)
HCA与解锁的CUDA任务的通信，消除了对CPU的依赖.CPU
仅需要准备和排队计算和通信任务。GPUDirect
Async的计算和发送工作流程见图2：

![](./media/image4.emf)

![](./media/image5.emf)

图2 GPUDirect Async的计算和发送流程图

使用GPUDirect
Async技术时，CPU工作荷载改变了。例如，在完成[准备和排队所有必要的任务]{.mark}到GPU上后，CPU可返回执行其他工作。

注意，GPUDirect Async与GPUDirect
RDMA[无关]{.mark}，因此可[单独]{.mark}执行测试。GPUDirect
RDMA计算效率严重依赖于PCIe结构，即PCIe桥连的类型和数目(。。。、X8、X16、连接NIC与GPU的switch以及GPU的架构。当通信数据量很大(large
message size)时，管线阶段[使用发送侧(on sender
side)的主机内存]{.mark}比使用GPUDirect RDMA效率更高。

### 3.1动机

对比下面没有和使用GPUDirect Async技术后，CPU的荷载变化。

![](./media/image6.emf)

图3 MPI多GPU应用程序时间线上的通信阶段

![](./media/image7.emf)

图4 多GPU应用程序时间线上的通信阶段（使用GPUDirect Async技术）

### 3.2实施

目前支持GPUDirect Async需要包含在MKNX OFED 4.0内的扩展的IB Verbs
API，以及最新版本的Mellanox Infiniband
HCA，libmlx5用户驱动程序支持这些API。

传统做法，CPU通过填充数据结构(在发送和接收内存队列)，提出与IB
HCA的通信操作（Work Requests or
WQEs），然后更新某种[门铃]{.mark}寄存器[(doorbell
register)]{.mark}，都与某种[队列对(Queue Pair,
QP)]{.mark}有关。需要更新门铃寄存器来提醒HCA新的requests准备好等待处理。在最新的Mellanox
HCA中，当启动发送操作，需要2个不同的门铃更新：一个到主机内存中的32位字(DBREC)，而另一个到位于一定偏移量的HW寄存器，进入HCA
PCIe资源(Base Address Register,
BAR)。当使用核函数支路时，用户空间进程，使用HCA
BAR寄存的锁定页的uncached memory-mapped IO
(MMIO)映射，直接更新门铃。当request完成后，即数据已经发送或接收，HCA向与创建时的QP相关的发送或接收Completion
Queue (CQ)，分别添加一个新的CQE(Completion Queue
Entry)。应用程序需要投票(poll)相应的CQ，以检测是否完成了request，见图5。

![](./media/image8.emf)

图5 Infiniband HCA的发送/接收要求的处理流程

当CPU仍然负责准备这些命令时，GPUDirect
Async使用2个CUDA驱动函数[cuMemHostRegister()和cuMemHostGetDevicePointer()]{.mark}，来要求GPU直接访问HCA门铃寄存器，直接访问CQs（驻留在主机内存中），cuMemHostRegister()页锁定当前的主机内存范围，并将其映射到GPU地址空间，而cuMemHostGetDevicePointer()取回相应的设备指针。

当注册属于第三方PCIe设备(即Infiniband
HCA)的MMIO地址时，使用CU_MEMHOSTREGISTER_IOMEMORY标记(flag)。PCIe设备对应所谓的GPU
peer mapping的创建，即GPU映射到peer
PCIe设备。注意，当前实施下，整个MMIO范围必须是物理上连续的，并标记对CPU缓存禁用。

由于在Pascal架构之前的NVIDIA GPU的HW限制，需要一种特殊的[Mellanox
HCA固件]{.mark}使HCA PCIe资源（BAR）能置于合适的地址范围。

Once the doorbell registers and the CQs are mapped on the GPU, it is
possible to access them on either (a) CUDA streams or (b) from CUDA
kernel threads. We refer to the former as the Stream Asynchronous (SA)
communication model and to the latter as the Kernel-Initiated (KI)
communication model.

### 3.3软件支持

为实施GPUDirect Async技术，需要在不同软件层级上实施和修改程序库，见图7。

![](./media/image9.emf)

图7 GPUDirect Async软件层级

#### 3.3.1 libibverbs

**libibverbs**实施OpenFabrics Infiniband Verbs
API。在4.0版本，Mellanox引入新的Peer-Direct
API（见peer_ops.h），服务NVIDIA GPUDirect Async技术。

#### 3.3.2 libmlx5

**libmlx5**是生产商提供的底层驱动，用于管理最新的Mellanox Infiniband
HCA。允许用户编写程序[低延迟(low latency)、低成本(low overhead)
（核函数支路，kernel by-pass)]{.mark}直接访问Mellanox HCA硬件。

#### 3.3.3 LibGDSync {#libgdsync .标题4}

本文作者开发，在Infiniband Verbs基础上实施GPUDirect
Async，连接CUDA与Verbs APIs。它由一系列底层API组成，与IB
Verbs非常相似，对CUDA流操作。LibGDSync负责创建Verbs对象，即QPs、CQs、GPUDirect
Async约束条件的结构体、注册需要的主机内存、附上发送指令和等待GPU流完成。函数像gds_stream_queue_send、gds_stream_wait_cq，内部使用CUDA流MemOp
API。

#### 3.3.4 LibMP

本文作者开发，是构建在LibGDSync
API基础上的消息库，用于在应用程序中部署GPUDirect Async技术。

当初始化MPI环境后（即communicator, ranks, topology,
etc.），可以用[LibMP]{.mark}
[API]{.mark}代替[标准的MPI通信]{.mark}原语，例如：用mp_isend_on_stream(
)代替MPI_Isend( )、用mp_wait_on_stream( )代替MPI_Wait( )等等。

![](./media/image10.emf)

![](./media/image11.emf)

当CPU post WQE，在收集descriptors以及将他们转换为CUDA
API调用，使用通信原语的参数（destination/source peer ranks, message
size, buffer pointers）。

具体编程看来还要看示例代码。

3.3.5系统要求

Async要求：

-   Mellanox [Connect-IB]{.mark} or later HCA, 可能要某种固件版本

-   MLNX OFED 4.0 for Peer-Direct Async Verbs APIs

-   CUDA 8.0 for Stream Memory Operations APIs

-   NVIDIA display driver version 384 or newer

-   LibGDSync library

-   A special NVIDIA kernel driver registry key is required to enable
    > GPU peer mappings

-   The nvidia_peer_memory kernel module

-   The [GDRcopy] library

[算法1]{.mark}展示了典型的GPUDirect
Async应用程序结构，使用LibMP函数，其中2个过程交换数据，使用SA模型，混合[通信和计算]{.mark}任务。

![](./media/image12.emf)

## 4 GPUDirect Async模型

LibMP包含2种不同的执行模型：[SA模型和KI模型]{.mark}。SA模型中，相对主机通信是[异步]{.mark}执行，与CUDA流[同步]{.mark}执行；KI模型中，在核函数中由CUDA线程启动通信。

本节采用[抽象效率]{.mark}模型，比较Async模型与标准MPI通信模型的行为。考察的执行流程是典型的GPU加速的MPI应用程序，其中各MPI
rank与其他peer (GPU)交替执行计算和通信。第6节将探讨GPUDirect
Async改进MPI模型效率的[条件]{.mark}。

### 4.1 CPU同步模型

[常规的多GPU
MPI应用]{.mark}，考虑D维迭代末班计算并行化的核函数，采用区域分解方法。需要3个独立的阶段：

**1、计算和发送(Compute and
Send)**：执行*X*次，启动（*LA~i~*次）一些CUDA任务（耗时*A~i~*），像核函数，或内存转移；在主机上执行一些操作，像与CUDA流同步（耗时*TH*），然后发送计算数据（耗时*S~i~*）。

**2、内部区域计算(Interior
Compute)**：执行*Y*次，在主机上执行一些操作（耗时*TH*），并启动（*LB~j~*次）一些CUDA任务（耗时*B~j~*）\--处理[内部数据单元]{.mark}（指当前计算节点内部）上，即与来自相邻[计算节点]{.mark}的数据无关。

**3、接收和计算(Receive and
Compute)**：执行*Z*次，等待接收来自其他进程的数据（*W~k~*），在主机上执行一些操作（耗时*TH*），并启动（*LC~k~*次）CUDA任务（耗时*C~k~*）\--处理接收数据。

考虑执行上述模式*R*次迭代，如图8，式（1）表示在CPU（![](./media/image13.wmf)）和GPU（![](./media/image14.wmf)）上的耗时，整个程序的耗时（*T~S~*）。

![](./media/image15.wmf) （1）

总耗时*T~S~*等于CPU耗时，因为CPU总是处于忙碌状态，即处理等待GPU任务完成的最糟糕情况，由![](./media/image16.wmf)参数表示：

![](./media/image17.wmf)

![](./media/image18.emf){width="5.657485783027122in"
height="2.1342125984251967in"}

图8 一个迭代的多GPU区域分解MPI应用程序的典型时间线

下面将考察LibMP的通信模型。

### 4.2 Stream asynchronous, SA模型

SA模型中，[通信与其他CUDA任务]{.mark}是一起排队进入CUDA流，像核函数、内存转移等。通常SA模型相对容易使用，因为只需要修改很少的MPI应用程序代码（即用mp_isend_on_stream修改MPI_Isend，忽略CUDA同步原语）。相对主机代码，计算和通信任务是[异步执行]{.mark}的，但与CUDA流是[同步的]{.mark}。

可以使用SA模型开发上述类型的应用程序，如果可以修改原始算法，使其符合下[式（2）和图9]{.mark}的流程：

![](./media/image19.wmf) （2）

式中，![](./media/image20.wmf)和![](./media/image21.wmf)分别表示在CPU上[排队发送和等待]{.mark}接收CUDA流上的操作的耗时。

SA模型中，由于异步行为，可以忽略![](./media/image22.wmf)耗时，CPU排队很多在CUDA流上顺序执行的任务，无需等待他们完成。

为确保异步行为，在通信期间要求：

（1）必须删除所有的CUDA同步原语；

（2）必须使用对应的CUDA异步原语替代所有非异步的CUDA原语；

（3）在posting时刻（如发送和接收buffer size, 目标rank,指针等），必须已知通信参数；

（4）所有的MPI函数必须用LibMP函数代替。

![](./media/image23.emf)

图9 多GPU的SA模型的一般通信模式，式（2）表征

一个明显的副作用就是：[CPU做更少的工作，]{.mark}因为主机代码即不做同步操作，也不执行通信，因此在异步的上下文环境下没有相关工作。因此，在式（2）中忽略TH参数。

与式（2）相符的算法表示在[异步模式]下效率改善[需要符合以下3种条件]

#### 4.2.1 C1条件：异步(Asynchronous)

式（2）中，总执行时间等于GPU时间，如果：

![](./media/image24.wmf)

就是[在CPU上排队任务到CUDA流的耗时（**启动时间**），必须低于在GPU上执行这些任务的耗时（C1条件）]{.mark}：

![](./media/image25.wmf)

如果不满足这个条件，异步就不能发生，因为CPU启动时间超过了GPU执行时间。

#### 4.2.2 C2条件：时间收益(time gain) {#c2条件时间收益time-gain .标题4}

SA模型（式2）比同步模型（式1）快，如果：

![](./media/image26.wmf)

满足上式的更严格的条件是：

![](./media/image27.wmf)

因为![](./media/image28.wmf)。

如果GPU计算任务在2种模型（同步模型和SA模型）中需要的时间一样，则：

![](./media/image29.wmf)

则，得到一个简单的条件：

![](./media/image30.wmf)

**意思是：**如果（SA模型的）[通信时间总和]{.mark}（发送*TS*和等待*TW*）低于（同步模型的）GPU的![](./media/image31.wmf)时间（GPU等待CPU工作的闲置时间），SA模型更快。

直觉上，[SA模型的效率依赖于]{.mark}SA模型的CUDA流同步通信与SA模型的CPU初始化通信加上S（同步）模型的GPU同步时间之和[的比值]{.mark}。

#### 4.2.3 C3条件：分片计算(fragmented computations) {#c3条件分片计算fragmented-computations .标题4}

子任务*R，X，Y，Z*的执行次数越多，执行异步的程度越高（因为C1）。

![](./media/image32.wmf)

### 4.3 Kernel-Initiated, KI模型

Streaming Multiprocessor
(SM)负责执行CUDA核函数，可直接使用通信原语发送消息或等待接收消息完成。GPU中有HCA门铃注册和CQ映射，一个CUDA线程可使用简单的值分配(value
assignment)和代码内部比较，分别响铃或poll
CQs。在KI模型中，使用核函数融合技术（[kernel fusion]{.mark}
technique），其中在一个单独的CUDA核函数中通信（发送或等待）和计算任务融合在一起。与GPUDirect
RDMA联合使用时，该方法会引起GPU内存一致性问题。为避免该问题，在测试算例中使用主机内存作为通信缓存。KI模型的典型时间线如图10。

![](./media/image33.emf){width="5.53805227471566in"
height="1.4543974190726159in"}

图10 KI通信模式时间线

与SA模型一样，CPU准备通信[描述子(descriptors)]{.mark}，之后直接由在CUDA核函数KI（使用描述子）中的线程直接启动通信，而不像SA模型中由CUDA流启动通信。复杂度移向CUDA核函数KI，需要至少N+M+1个[块(blocks)]{.mark}，其中N是在发送操作之前执行计算A类型任务需要的块数，M是处理接收数据执行C类任务需要的块数，加1是用来poll
CQs，见图11。与SA模型一样，B任务代表与通信无关的其他（可能的）CUDA任务，由某种类型的块实施，块与具体算法有关。

![](./media/image34.emf){width="5.102372047244095in"
height="3.389469597550306in"}

图11 KI核函数，CUDA块任务

在KI模型中，核函数融合技术与动态调度程序联合使用，使用原子操作pick各线程块（不能保证线程块被GPU
HW安排的顺序），然后根据如下原则分配到正确的任务：

（1）为避免死锁，receive操作必须从一开始不能阻止send操作或继续通信：必须总有（至少）一个等待接收的线程块，其他线程块执行send。

（2）Receive时间很关键，因此使用第一个receiver线程块等待进入的消息。各线程poll与各远程计算节点相关的接收CQ。

（3）第2个到第N+1个线程块被分配到A操作组，而剩余的M个线程块分配到C操作组，等待receiver线程块，发出信号：已经接收所有进入的数据。

（4）使用inter-block
barrier方法同步receiver和C任务的线程块，见图12，其中[各C任务线程块]{.mark}的0号线程等待receiver线程块的0号线程[将全局内存变量设为1]{.mark}（号线程？），而剩余的线程移向\_\_syncthreads()
barrier。为防止等待receiver的CUDA线程块的浪费，在执行C任务之前执行A任务，B任务和send。

![](./media/image35.emf){width="4.736111111111111in"
height="3.249860017497813in"}

图12 线程块之间的同步(barrier)

（5）当上述通信发生（在接收完成之后），在C任务线程块中的所有线程0将到达\_\_syncthreads()
barrier，然后开始[解包]{.mark}接收的数据。

KI模型的执行时间可评估如下：

![](./media/image36.wmf) （3）

式中，![](./media/image37.wmf)是为了执行A任务加上send和sender线程块的耗时；![](./media/image38.wmf)是等待数据和执行C任务的耗时；![](./media/image39.wmf)代表其他与通信无关的任务（像处理内部数据），这可以由任何类型的线程块执行（A线程块、C线程块或其他线程块）。

测量得到：![](./media/image40.wmf)总是可忽略。因此，可认为![](./media/image41.wmf)。

当在GPU上运行时，一个CUDA核函数的多个线程块都并发执行，因此执行任务A+S，任务W+C和任务B可以重叠。为表示重叠程度，定义Overlap，表示考虑若干输入参数（像GPU的SM、任务分配算法、具体的通信模式、A，B，C任务的计算时间）后的所有任务间的重叠时间，这是一个复杂的度量。

相对SA模型，KI模型的时间收益可表述为：

![](./media/image42.wmf)

最佳工况是：当所有任务[适应(fit
in)]{.mark}GPU上可获取的CUDA线程块的逻辑个数和大部分执行任务的重叠：

![](./media/image43.wmf)

相反，最糟糕工况是：当各任务需要最大数目的线程块，降低了重叠时间的重要性：

![](./media/image44.wmf)

这意味着A，B，C任务几乎是[顺序执行]{.mark}，与SA模型一样，式（3）中的重叠时间表示相对式（2），不能有实际的效率改善。

## 5 Ping-Pong测试

ping-pong延迟测试，3种模型：标准MPI，SA，KI，基于点对点(send-receive)通信。

### 5.1 ping-pong延迟测试

不考虑GPU核函数计算，仅评估通信耗时。

算法2描述了2个MPI进程（rank0和rank1）之间的简单ping-pong测试，交换位于主机上的内存，选择性地执行常数时间的CUDA核函数。

![](./media/image45.emf)

如图13，小的通信信息大小时，标准MPI更快，因为GPU通信路径的overhead。随着信息大小增加，延迟线性增长，但SA模型的延迟比MPI和KI模型的更不规则，[SA模型的延迟Piece-wise
constant periods interleaved with sudden
peaks]{.mark}，这需要解释CUDA流是如何执行通信的。

![](./media/image46.emf)

图13 标准MPI，SA和KI模型ping-pong测试（仅考虑通信），Kepler架构

接下来，详细测试并解释了SA模型延迟的[Piece-wise constant periods
interleaved with sudden peaks]{.mark}的原因：[polling pattern of the GPU
front-end unit]{.mark}。

并在更新的Maxwell架构（延迟相似）和Pascal架构（Tesla卡，延迟更好）的GPU上执行。

### 5.3 带GPU计算的ping-pong延迟

图18中绘制了双程(Round-trip---数据在主机和设备间的拷贝)延迟，此时启动了\~5微秒的CUDA核函数计算。整体计算效率与计算和通信都有关系：

![](./media/image47.emf)

图18标准MPI，SA和KI模型ping-pong测试（通信+计算）

使用Nvidia visual profiler工具分析。

（1）MPI：rank0等待接收数据完成，启动核函数，与CUDA流同步，最终发送数据。

（2）SA模型：GPU与CPU有很大的重叠。

（3）KI模型：仅有一系列的核函数在CUDA流上执行。

KI模型效率最高。考虑核函数计算后，没有出现[Piece-wise constant periods
interleaved with sudden peaks]{.mark}的现象。

## 6 应用程序测试

### 6.1 HPGMG-FV CUDA

HPGMG-FV CUDA的主要通信函数在GPU层级上[服从2D
Stencil模式]{.mark}，与4.1节的CPU同步模式类似：

**1、Pack**：单独的CUDA核函数，其中存储了结果数据，进入发送缓冲区（A类任务）；

2、Send: 与CUDA流同步，发送结果数据；

3、Interior computation: 单独的核函数处理内部结构（B类任务）；

4、Receive：从其他进程接收数据；

5、Unpack：单独的核函数计算处理接收的数据（C类任务）。

算法3是以上通信函数的MPI伪代码。

![](./media/image48.emf)

通信模式示意图见图21，CUDA Visual
Profiler分析见图22：在CUDA核函数之间，GPU为加载，一直等待从CPU的新任务（绿色IDLE区域）。

![](./media/image49.emf)

图21 HPGMG-FV的通信函数时间线，MPI版本

![](./media/image50.emf)

图22 CUDA Visual Profiler上的HPGMG-FV通信函数，MPI版本

下面应用SA和KI模型修改HPGMG-FV CUDA算法。

#### 6.1.1 SA(Stream Asynchronous)模型

CUDA核函数与通信之间的主机代码是在Pack核函数之后，简单的cudaDeviceSynchronize()组成，可删去（忽略TH）。另外，通信参数在启动通信时已知，因此可实施SA模型。

修改算法3中的[exchangeLevelBoundariesMPI]{.mark}为算法4中的[exchangeLevelBoundariesSA]{.mark}。在Infiniband通信的情况下，如果启动send，但对应的receive尚未准备好，则通信将存在延迟。为此，我们使用one-sided异步调用[mp_iput_on_stream]{.mark}，来确保各[异步send]{.mark}有相应的其他peer进程发出的receive缓冲。

![](./media/image51.emf)

如图25，Y轴显示了SA模型下，GPU层级相对标准MPI版本的效率增加。

![](./media/image52.emf)

图25
相对MPI版本，实施SA模式的HPGMG-FV的时间收益，仅比较GPU层，直到16个进程，weak-scaling

使用2个进程，在log~2~(size)=4工况下，获得最佳收益（约24%），当增大计算规模，计算效率收益下降，原因有二：

（1）计算规模增大后，通信[消息大小增大，（由于计算时间增长），通信成本变的不重要]{.mark}。当计算规模很大时，所有的通信方式都趋向于相同的效率水平。

（2）增加计算规模，增加CUDA核函数计算负荷（即计算时间），降低了GPU闲置时间。因此，用CUDA流上的通信代替短暂的GPU闲置时间，不能显著改善计算效率（C2条件）。并且，如果[通信时间超过GPU闲置时间]{.mark}还会降低计算效率。

#### **我的理解**

对第1点：通信信息量随着计算规模增大而增大，通信成本反而变的不重要？？？（计算部分的比重提高了，使通信比例下降）。通信量比重较大时，实施异步通信模式是很有益的！

对第2点：计算规模增大，GPU的计算负荷增大，GPU闲置时间降低。所以，要想异步通信有积极的时间收益，必须使得在GPU闲置空挡内完成通信。

看来要是Async促进效率，要平衡通信量与计算量：低阶格式的通信量相对计算量较小，当计算量很大时，如果降低了通信所占的比例，Async通信提高计算效率的效果会降低。同等计算规模下，高阶格式的通信量更大；而高阶格式一般在相对粗网格上执行，GPU的计算量较低阶格式要低，即GPU闲置时间期望得到延长，可对付通信时间增长的问题（但也要将通信时间控制在GPU闲置时间以内）。因此，Async通信模式对解决高阶格式的通信overhead问题期望有缓解的作用。

#### 6.1.2 KI(Kernel-Initiated)模型

根据之前的观察，Pack核函数是A类型任务，Unpack核函数是C类型任务，Interior核函数是B类型任务。因此，使用如图26所示的组织CUDA线程块的单独核函数，修改各通信阶段。

![](./media/image53.emf)

图26 HPGMG-FV通信模式，KI版本

B类型任务是在发送操作后，由A类型任务计算的，为了与Wait和Unpack操作重叠。

SA模型中，需要的CUDA线程块（Pack, Interior,
Unpack）总和是193个，每个有64个线程。各线程需要约37个寄存器，无需共享内存。Tesla
K20 GPU拥有13
SMs，各SM最多有2048线程，这意味着所有的193个逻辑CUDA线程块可以并发执行，所有任务在大部分时间可重叠，这是最佳的KI方案（4.3节）。观察KI模型的Visual
Profiler，得到减少的TCPU~KI~和TGPU~KI~时间，见表2.

![](./media/image54.emf)

启动2个进程时获得最佳计算收益，约26%，KI模型的计算效率比SA模型更好。

### 6.2 CoMD-CUDA 

分子动力学模型，SA模型与KI模型的计算效率收益[类似]{.mark}，Async都有积极作用。

### 6.3 BFS

SA模型没有改善计算效率。

### 6.4 LULESH2-CUDA

Livermore Unstructured Lagrange Explicit Shock Hydrodynamics (LULESH)
proxy application at Lawrence Livermore National Laboratory (LLNL).

LULESH2是将结构网格转换为非结构网格编号实施计算的。在E2环境下，使用27个计算节点，采用60^3^规模的结构网格，增加循环次数，加强GPU计算荷载。只实施了SA模型。SA模型相对MPI模型的时间收益见下图。

![](./media/image55.emf)

图 LULESH2-CUDA时间收益，SA模型，27个进程（MPI）

## 7 结论和展望

底层：Mellanox OFED 4.0

中间层：LibGDSync

顶层：LibMP

（1）GPUDirect Async构建了多GPU加速应用的[新的通信]}模型；

（2）[GPUDirect Async不一定更快]

（3）CPU能启动更多[若干连续的异步]通信周期（MPI_barrier()），获得的加速效果越好。

GPUDirect Async的主要缺陷：

（1）通信参数必须在posting on GPU Stream之前已知；

（2）如果GPU过载，或者如果[(GPU)]{.mark}闲置时间小于通信时间，计算效率实际是下降的（换句话说，就是如果Async技术有效，得使GPU闲置时间略大于通信时间）。

## 参考文献

E. Agostini, D. Rossetti, S. Potluri. 2018. GPUDirect Async: Exploring
GPU synchronous communication techniques for InfiniBand clusters. J.
Parallel Distrib. Comput. 114: 28-45.

A. Venkatesh, K. Hamidouche, S. Potluri, D. Rossetti, C.H. Chu, D.K.
Panda, MPI-GDS: High performance MPI designs with GPUDirect-aSync for
CPU-GPU control flow decoupling, in: International Conference on
Parallel Processing, August 2017.
（[GPUDirect-aSync技术中在CPU执行通信，不能像本文一样独立执行]{.mark}）

N. Sakharnykh, High-Performance Geometric Multi-Grid with GPU
Acceleration.https://devblogs.nvidia.com/parallelforall/high-performance-geometric-multi-grid-gpu-acceleration.

E. Agostini, D. Rossetti, S. Potluri. 2017. Offloading communication
control logic in GPU accelerated applications. 2017 17th IEEE/ACM
International Symposium on Cluster, Cloud and Grid Computing.
