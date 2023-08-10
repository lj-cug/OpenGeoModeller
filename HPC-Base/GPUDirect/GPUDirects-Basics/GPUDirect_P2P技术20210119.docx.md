# 设备间通信

## cparallel_mpi.c

包含：

MPI初始化（计算进程与IO进程分开）

阻塞式进程数据通信：MPI_Send,
MPI_Recv，[等价于SUNTANS的sendrecv.c程序。]{.mark}

## cparallel_mpi_gpu.cu

有如下数据交换函数API，但是都没有被调用过，见csolver_gpu.cu。多种通信方式[，包括6种]{.mark}：

[exchange2d_mpi_gpu]{.mark} [// 一般的MPI-CUDA通信]{.mark}

[exchange3d_mpi_gpu]{.mark}

[exchange2d_cuda_aware_mpi]{.mark} [// CUDA-aware MPI通信]{.mark}

[exchange3d_cuda_aware_mpi]{.mark}

[exchange2d_cuda_ipc]{.mark}

[exchange3d_cuda_ipc]{.mark}

[exchange2d_cudaPeer]{.mark}

[exchange3d_cudaPeer]{.mark}

[exchange2d_cudaPeerAsync]{.mark}

[exchange3d_cudaPeerAsync]{.mark}

[exchange2d_cudaUVA]{.mark}

[exchange3d_cudaUVA]{.mark}

还有一些施加周期边界条件的API，如xperi3d_mpi_gpu、yperi3d_mpi_gpu、。。。

### [exchange2d_mpi_gpu]{.mark} {#exchange2d_mpi_gpu .标题3}

[float h_work\[j_size\]\[i_size\];]{.mark} //
主机上开辟一个临时用于主机和设备间转数据的数组

[(1) cudaMemcpy(h_work, d_work, nx\*ny\*sizeof(float),
cudaMemcpyDeviceToHost);
//将设备上的数组d_work，传输到主机上的h_work]{.mark}

[(2) exchange2d_mpi(h_work, nx, ny);]{.mark} // 然后，MPI通信交换数据

[(3) checkCudaErrors(cudaMemcpy(d_work, h_work, nx\*ny\*sizeof(float),
cudaMemcpyHostToDevice)); // 交换后的数组再传输到设备上d_work]{.mark}

### [exchange2d_cuda_aware_mpi]{.mark} {#exchange2d_cuda_aware_mpi .标题3}

gpuPOM采用的是非阻塞式通信

以east方向为例（因为是结构网格，分2个方向）：

[float \*d_send_to_east = d_1d_ny_tmp0;]{.mark}

[float \*d_recv_from_east = d_1d_ny_tmp3;]{.mark}

[MPI_Request request\[2\];]{.mark}

[MPI_Status status\[2\];]{.mark}

//
设备到设备的传输数据，到指针[d_send_to_east]{.mark}（待发送到东边的数据）

[checkCudaErrors(cudaMemcpy2D(d_send_to_east, sizeof(float),]{.mark}

[ d_work+(nx-2), nx\*sizeof(float),]{.mark}

[ sizeof(float), ny,]{.mark}

[ cudaMemcpyDeviceToDevice));]{.mark}

[MPI_Isend(d_send_to_east, ny, MPI_FLOAT, n_east, my_task,]{.mark}

[ pom_comm, &request\[0\]); // 发送到east]{.mark}

[MPI_Irecv(d_recv_from_east, ny, MPI_FLOAT, n_east, n_east,]{.mark}

[ pom_comm, &request\[1\]); // 接收来自east]{.mark}

[MPI_Waitall(2, request, status);]{.mark} //同步（非阻塞通信必须的步骤）

// 最后，设备到设备的传输数据，来自east的数据

[checkCudaErrors(cudaMemcpy2D(d_work, nx\*sizeof(float),
d_recv_from_east, sizeof(float), sizeof(float), ny,
cudaMemcpyDeviceToDevice));]{.mark}

### [exchange2d_cuda_ipc（IPC: Inter Process Communication）]{.mark} {#exchange2d_cuda_ipcipc-inter-process-communication .标题3}

仅支持64位Linux系统。

Using this API, an application can get the IPC handle for a given device
memory pointer using cudaIpcGetMemHandle(), pass it to another process
using standard IPC mechanisms (e.g., interprocess shared memory or
files), and use cudaIpcOpenMemHandle() to retrieve a device pointer from
the IPC handle that is a valid pointer within this other process. Event
handles can be shared using similar entry points.

[请参考cuda_samples/SimpleIPC的示例代码]{.mark}

要仔细研究！

[void exchange2d_cuda_ipc(float \*d_send,]{.mark}

[ float \*d_east_recv,]{.mark}

[ float \*d_west_recv,]{.mark}

[ cudaStream_t &stream_in,]{.mark}

[ int nx, int ny){]{.mark}

1.  [MPI_Barrier(pom_comm);]{.mark}

[2 checkCudaErrors(cudaMemcpy2DAsync(d_east_recv,
nx\*sizeof(float),]{.mark}

[ d_send+(nx-2), nx\*sizeof(float),]{.mark}

[ sizeof(float), ny,]{.mark}

[ cudaMemcpyDefault,]{.mark}

> [ stream_in));]{.mark}
>
> [cudaStreamSynchronize]{.mark}

2.  [MPI_Barrier(pom_comm);]{.mark}

gpuPOM模型中使用IPC通信方式的相关函数：

[cudaIpcMemHandle_t handle_elf]{.mark}

[void exchangeMemHandle()]{.mark}： [cudaIpcGetMemHandle]{.mark}

[void openMemHandle()]{.mark}

### [exchange2d_cudaPeer]{.mark}（non-UVA通信方式） {#exchange2d_cudapeernon-uva通信方式 .标题3}

[cudaMemcpy3DPeerParms p_east_recv={0};]{.mark}

[p_east_recv.extent = make_cudaExtent(sizeof(float), ny, nz);]{.mark}

[p_east_recv.dstDevice = n_east;]{.mark}

[p_east_recv.dstPtr = make_cudaPitchedPtr(d_east_recv,
nx\*sizeof(float), nx, ny);]{.mark}

[p_east_recv.srcDevice = my_task;]{.mark}

[p_east_recv.srcPtr = make_cudaPitchedPtr(d_send+(nx-2),
nx\*sizeof(float), nx, ny);]{.mark}

[//! send ghost cell data to the east]{.mark}

[checkCudaErrors(cudaMemcpy3DPeerAsync(&p_east_recv,
stream_in));]{.mark}

[checkCudaErrors(cudaStreamSynchronize(stream_in));]{.mark}

### [exchange2d_cudaPeerAsync]{.mark}（non-UVA通信方式） {#exchange2d_cudapeerasyncnon-uva通信方式 .标题3}

[跟exchange2d_cudaPeer的代码一样。得仔细研究一下。]{.mark}

### [exchange3d_cudaUVA（当前使用的UVA通信方式）]{.mark} {#exchange3d_cudauva当前使用的uva通信方式 .标题3}

**[UVA: Unified Virtual Adress]{.mark}**

[(1)MPI_Barrier(pom_comm);]{.mark}

[//! send ghost cell data to the east]{.mark}

[//! recieve ghost cell data from the west]{.mark}

[(2) cudaMemcpy2DAsync]{.mark}

[//! send ghost cell data to the north]{.mark}

[//! recieve ghost cell data from the south]{.mark}

[(3) cudaMemcpy2DAsync]{.mark}

[(4) checkCudaErrors(cudaStreamSynchronize(stream_in));]{.mark}

[(5) MPI_Barrier(pom_comm);]{.mark}

### 多GPU的P2P复制和访问 {#多gpu的p2p复制和访问 .标题3}

Peer-to-peer memcpy，GPU A上的指针A直接数据复制到GPU B上的指针B

（1）使用UVA（统一虚拟地址）

仅使用cudaMemcpy(..., cudaMemcpyDefault)或者

cudaMemcpyAsync(..., cudaMemcpyDefault)

（2）使用non-UVA[显式地]{.mark}做P2P复制

cudaError_t cudaMemcpyPeer( void \* dst, int dstDevice, const void\*
src,

int srcDevice, size_t count )

cudaError_t cudaMemcpyPeerAsync( void \* dst, int dstDevice,

const void\* src, int srcDevice, size_t count, cuda_stream_t stream = 0
)

示例代码：

（1）当可使用UVA时，可使用cudaMemcpy用于peer-to-peer复制内存，因为CUDA可以推测设备"拥有"它自己的内存。实施代码如下：

//Check for peer access between participating GPUs:

cudaDeviceCanAccessPeer(&can_access_peer_0_1, gpuid_0, gpuid_1);

cudaDeviceCanAccessPeer(&can_access_peer_1_0, gpuid_1, gpuid_0);

//Enable peer access between participating GPUs:

cudaSetDevice(gpuid_0);

cudaDeviceEnablePeerAccess(gpuid_1, 0);

cudaSetDevice(gpuid_1);

cudaDeviceEnablePeerAccess(gpuid_0, 0);

//UVA memory copy:

cudaMemcpy(gpu0_buf, gpu1_buf, buf_size, cudaMemcpyDefault);

（2）当没有UVA功能时，通过cudaMemcpyPeer执行peer-to-peer复制内存。

// Set device 0 as current

cudaSetDevice(0);

float\* p0;

size_t size = 1024 \* sizeof(float);

// Allocate memory on device 0

cudaMalloc(&p0, size);

// Set device 1 as current

cudaSetDevice(1);

float\* p1;

// Allocate memory on device 1

cudaMalloc(&p1, size);

// Set device 0 as current

cudaSetDevice(0);

// Launch kernel on device 0

MyKernel\<\<\<1000, 128\>\>\>(p0);

// Set device 1 as current

cudaSetDevice(1);

// Copy p0 to p1

cudaMemcpyPeer(p1, 1, p0, 0, size);

// Launch kernel on device 1

MyKernel\<\<\<1000, 128\>\>\>(p1);

cudaMemcpyAsync：不同Stream之间传输数据使用的API，该函数在主机端是非阻塞的，传输处理后，控制权马上返回给主机线程。

cudaMemcpypeerAsync：cuda提供[cudaMemcpyPeerAsync]{.mark}实现显卡间的数据复制，但是该函数需要硬件支持。可以使用cuda安装程序提供的p2pBandwidthLatencyTest测试当前硬件是否支持直接在显卡间实现数据复制。

### 总结 {#总结 .标题3}

-   带Async的API函数，都涉及多流之间的数据交换。

-   带Peer的API，就是non-UVA，不需要cudaMallocHost(页锁定内存)；

-   UVA方式：不带Peer，一般的cudaMemcpy用于多GPU间数据复制时，需要启动PeerAccess，即cudaDeviceEnablePeerAccess，然后cudaMemcpy使用参数cudaMemcpyDefault或cudaMemcpyDevicetoDevice
