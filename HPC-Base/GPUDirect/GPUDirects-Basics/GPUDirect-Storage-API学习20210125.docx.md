# GPUDirect Storage API使用

[注意：]{.mark}

cuFile API是线程安全的。

初始化GDS库后，不应使用fork系统调用。fork系统调用后API的行为在子进程中没有定义。API的设计不与fork调用工作。

应在有效的CUDA上下文环境中调用带GPU缓存的API。

## 1 [cuFileDriverOpen]{.underline}

在启用任何其他GDS
API之前，各进程仅激活一次该API。应用程序应调用该API来避免在第一次IO调用时的驱动程序的延迟。

## [2. cuFileHandleRegister]{.underline}

该API将文件描述词转换为cuFileHandle，检查在挂载点上的命名文件的能力，在该平台上通过GDS得到支持。

[注意：]{.mark}各文件描述词应有一个句柄。

多线程共享相同句柄。查看示例代码展示使用多个线程使用相同句柄的信息。

[注意：]{.mark}在兼容模式，无需O_DIRECT模式可打开另外的fd。该模式还可以处理unaligned的读和写，甚至当POSIX无法处理时。

## [3. cuFileBufRegister, cuFileRead, and cuFileWrite]{.underline}

GPU内存应暴露给第三方设备来启动被这些设备使用的DMA。页面跨越这些在GPU虚拟地址空间中的缓冲，需要映射到BAR空间，该映射有一个overhead。

注意：实现该映射的进程称之为注册(registration)。

使用[cuFileBufRegister]{.underline}显式执行缓冲注册是可选的。如果没有注册用户缓冲，使用cuFile实施的一个中间的预注册的GPU缓冲，从该缓冲复制到用户缓冲。下表列出是否注册有益的参考。

注意：IO Pattern 1是不优化的baseline情况，不在此表中。

+----------------+-----------------------+-----------------------------+
| **Use Case**   | **Description**       | **Recommendation**          |
+================+=======================+=============================+
| A 4KB-aligned  | The GPU buffer is     | Register this reusable      |
| GPU buffer is  | used as an            | intermediate buffer to      |
| reused as an   | intermediate buffer   | avoid the additional        |
| intermediate   | to stream the         | internal staging of data by |
| buffer to read | contents or to        | using GPU bounce buffers in |
| or write data  | populate a different  | the cuFile library.         |
| by using       | data structure in GPU |                             |
| optimal IO     | memory.               | See [[IO Pattern            |
| sizes for      |                       | 2]{.underline}](https://do  |
| storage        | You can implement     | cs.nvidia.com/gpudirect-sto |
| systems in     | this use case for IO  | rage/best-practices-guide/i |
| multiples of   | libraries with DSG.   | ndex.html#io-pattern-2) for |
| 4KB.           |                       | the recommended usage.      |
+----------------+-----------------------+-----------------------------+
| Filling a      | The GPU buffer is the | This can also cause BAR     |
| large GPU      | final location of the | memory exhaustion because   |
| buffer for one | data. Since the       | running multiple threads or |
| use.           | buffer will not be    | applications will compete   |
|                | reused, the           | for BAR memory.             |
|                | registration cost     |                             |
|                | will not be           | Read or write the data      |
|                | amortized. A usage    | without buffer              |
|                | example is reading    | registration.               |
|                | large preformatted    |                             |
|                | checkpoint binary     | See [[IO Pattern            |
|                | data.                 | 3]{.underline}](https://do  |
|                |                       | cs.nvidia.com/gpudirect-sto |
|                | Registering a large   | rage/best-practices-guide/i |
|                | buffer can have a     | ndex.html#io-pattern-3) for |
|                | latency impact when   | the recommended usage.      |
|                | the buffer is         |                             |
|                | registered.           |                             |
+----------------+-----------------------+-----------------------------+
| Partitioning a | The main thread       | Allocate, register, and     |
| GPU buffer to  | allocates a large     | deregister the buffers in   |
| be accessed    | chunk of memory and   | each thread independently   |
| across         | creates multiple      | for simple IO workflows.    |
| multiple       | threads. Each thread  |                             |
| threads.       | registers a portion   | For cases where the GPU     |
|                | of the memory chunk   | memory is preallocated,     |
|                | independently and     | each thread can set the     |
|                | uses that as in [[IO  | appropriate context and     |
|                | Pattern               | register the buffers        |
|                | 2]{.underline}](http  | independently.              |
|                | s://docs.nvidia.com/g |                             |
|                | pudirect-storage/best | See IO Pattern 6 for the    |
|                | -practices-guide/inde | recommended usage.          |
|                | x.html#io-pattern-2). |                             |
|                |                       | After you install the GDS   |
|                | You can also register | package,                    |
|                | the entire memory in  | see cuf                     |
|                | the parent thread and | ile_sample_016.cc and cufil |
|                | use this registered   | e_sample_017.cc under /usr/ |
|                | buffer with the size  | local/CUDA-X.y/samples/ for |
|                | and dev               | more details.               |
|                | Ptr_offset parameters |                             |
|                | set appropriately     |                             |
|                | with the buffer       |                             |
|                | offsets for each      |                             |
|                | thread. A cudaContext |                             |
|                | must be established   |                             |
|                | in each thread before |                             |
|                | registering the GPU   |                             |
|                | buffers.              |                             |
+----------------+-----------------------+-----------------------------+
| GPU offsets,   | The IO reads or       | **Do not** register the     |
| file offsets,  | writes are mostly     | buffer.                     |
| and IO request | unaligned. An         |                             |
| sizes are      | intermediate aligned  | See [[IO Pattern            |
| unaligned.     | buffer might be       | 4]{.                        |
|                | needed to handle      | underline}](https://docs.nv |
|                | alignment issues with | idia.com/gpudirect-storage/ |
|                | GPU offsets, file     | best-practices-guide/index. |
|                | offsets, and IO       | html#io-pattern-4) and [[IO |
|                | sizes.                | Pattern                     |
|                |                       | 5]{.underline}](https:/     |
|                |                       | /docs.nvidia.com/gpudirect- |
|                |                       | storage/best-practices-guid |
|                |                       | e/index.html#io-pattern-5). |
+----------------+-----------------------+-----------------------------+
| Working on a   | In some GPU SKUs, the | To avoid failures because   |
| GPU with a     | BAR memory is smaller | of BAR memory exhaustion,   |
| small BAR      | than the total device | do not register the buffer. |
| space as       | memory.               |                             |
| compared to    |                       | See [[IO Pattern            |
| the available  |                       | 3]{.underline}](https:/     |
| GPU memory.    |                       | /docs.nvidia.com/gpudirect- |
|                |                       | storage/best-practices-guid |
|                |                       | e/index.html#io-pattern-3). |
+----------------+-----------------------+-----------------------------+

通过O_DIRECT方式打开文件，同时在IO时要求块大小对齐。

[aligned IO的性能要比unaligned
IO的性能好很多](http://www.mysqlperformanceblog.com/2011/06/09/aligning-io-on-a-hard-disk-raid-the-theory/)。

## [4. cuFileHandleDeregister]{.underline}

要求：在调用该API之前，应用程序必须确保已经完成IO在句柄上，不在被使用。文件描述词应该打开状态。总是使用该API，在结束进程之前重申资源。

## 5、[[cuFileBufDeregister]{.underline}](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html#cufile-bug-deregister)

要求：

## 6、[[cuFileDriverClose]{.underline}](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html#cufile-driver-close)

要求：

# cuFile APIs

cuFile
API是对CUDA新的扩展，因此需要关注将来API的变化。例如，一些领域还在审查，包括cuFileDriver属性和cuFileBuf管理。

## 1 使用

### 1.1动态交互 {#动态交互 .标题3}

Some of the cuFile APIs are optional, and if they are not called
proactively, their actions will occur reactively:

if cuFile{Open, HandleRegister, BufRegister} are called on a driver,
file, or buffer, respectively that has been opened or registered by a
previous cuFile\* API call, results in an error. Calling
cuFile{BufDeregister, HandleDeregister, DriverClose} on a buffer, file,
or driver, respectively that has never been opened or registered by a
previous cuFile\* API call results in an error. For these errors, the
output parameters of the APIs are left in an undefined state, but there
are no other side effects.

-   cuFileDriverOpen 显式地初始化设备驱动

Its use is optional, and if it is not used, driver initialization
happens implicitly at the first use of the cuFile{HandleRegister, Read,
Write, BufRegister} APIs.

-   cuFileHandleRegister

turns an OS-specific file descriptor into a cuFileHandle and performs
checking on the GDS supportability based on the mount point and the way
that the file was opened.

**Note:** Calling this API is a hard requirement.

-   cuFileBufRegister explicitly registers a memory buffer.

If this API is not called, a memory buffer is registered the first time
the buffer is used, for example, in cuFile{Read, Write}.

-   cuFile{BufDeregister, HandleDeregister} explicitly frees a buffer
    and file resources.

If this API is not called, the buffer and resources are implicitly freed
when the driver is closed.

-   cuFileDriverClose explicitly frees driver resources.

If this API is not called, the driver resources are implicitly freed
when the process is terminated.

If cuFile{Open, HandleRegister, BufRegister} is called on a driver,
file, or buffer, respectively that has been opened or registered by a
previous cuFile\* API call, results in an error.
Calling cuFile{BufDeregister, HandleDeregister, DriverClose} on a
buffer, file, or driver, respectively that has never been opened or
registered by a previous cuFile\* API call also results in an error. For
these errors, the output parameters of the APIs are left in an undefined
state and there are no other side effects.

### 1.2驱动，文件和缓冲器管理 {#驱动文件和缓冲器管理 .标题3}

下面是管理驱动，文件和缓冲器的整体流程：

1.  Call cuFileDriverOpen() to initialize the state of the critical
    performance path.

2.  Allocate the GPU memory with cudaMalloc.

3.  To register the buffer, call cuFileBufRegister to initialize the
    buffer state of the critical performance path.

4.  Complete the following IO workflow:

    a.  For Linux, open a file with POSIX open.

    b.  Call cuFileHandleRegister to wrap an existing file descriptor in
        an OS-agnostic cuFileHandle. This step evaluates the suitability
        of the file state and the file mount for GDS and initializes the
        file state of the critical performance path.

    c.  Call cuFileRead/cuFileWrite on an existing cuFile handle and
        existing buffer.

        -   If the cuFileBufRegister has not been previously called, the
            first time that cuFileRead/cuFileWrite is accessed, the GDS
            library performs a validation check on the GPU buffer and an
            IO is issued.

        -   Not using cuFileBufRegister might not be performant for
            small IO sizes.

        -   Refer to the [[GPUDirect Best Practices
            Guide]{.underline}](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html) for
            more information.

    d.  Unless an error condition is returned, the IO is performed
        successfully.

5.  Call cuFileBufDeregister to free the buffer-specific cuFile state.

6.  Call cuFileHandleDeregister to free the file-specific cuFile state.

7.  Call cuFileDriverClose to free up the cuFile state.

## 2 cuFile API定义

### 2.1数据类型

首先，cuFile
API使用数据类型，然后typedef使用数据类型，最后枚举也使用数据类型。

### 2.2 cuFile驱动API

用来初始化、终止、查询和调整设置cuFile系统的API：

/\* Initialize the cuFile infrastructure \*/

CUfileError_t cuFileDriverOpen();

/\* Finalize the cuFile system \*/

CUfileError_t cuFileDriverClose();

/\* Query capabilities based on current versions, installed
functionality \*/

CUfileError_t cuFileGetDriverProperties(CUfileDrvProps_t \*props);

/\*API to set whether the Read/Write APIs use polling to do IO
operations \*/

CUfileError_t cuFileDriverSetPollMode(bool poll, size_t
poll_threshold_size);

/\*API to set max IO size(KB) used by the library to talk to nvidia-fs
driver \*/

CUfileError_t cuFileDriverSetMaxDirectIOSize(size_t max_direct_io_size);

/\* API to set maximum GPU memory reserved per device by the library for
internal buffering \*/

CUfileError_t cuFileDriverSetMaxCacheSize(size_t max_cache_size);

/\* Sets maximum buffer space that is pinned in KB for use by
cuFileBufRegister \*/

CUfileError_t cuFileDriverSetMaxPinnedMemSize(size_t
max_pinned_memory_size);

### 2.3 cuFile IO APIs

The core of the cuFile IO APIs are the read and write functions.

[ssize_t cuFileRead(CUFileHandle_t fh, void \*devPtr_base, size_t size,
off_t file_offset, off_t devPtr_offset);]{.mark}

This API reads the data from a specified file handle at a specified
offset and size bytes into the GPU memory by using GDS functionality.
The API works correctly for unaligned offsets and any data size,
although the performance might not match the performance of aligned
reads.This is a synchronous call and blocks until the IO is complete.

Note: For the devPtr_offset, if data will be read starting exactly from
the devPtr_base that is registered with cuFileBufRegister, devPtr_offset
should be set to 0. To read starting from an offset in the registered
buffer range, the relative offset should be specified in the
devPtr_offset, and the devPtr_base must remain set to the base address
that was used in the cuFileBufRegister call.

[ssize_t cuFileWrite(CUFileHandle_t fh, const void \*devPtr_base, size_t
size, off_t file_offset, off_t devPtr_offset);]{.mark}

This API writes the data into a specified file handle at a specified
offset and size bytes from the GPU memory by using GDS functionality.
The API works correctly for unaligned offset and data sizes, although
the performance is not on-par with aligned writes.This is a synchronous
call and will block until the IO is complete.

If the file is opened with an O_SYNC flag, the metadata will be written
to the disk before the call is complete.

The buffer on the device has both a base (devPtr_base) and offset
(devPtr_offset). This offset is distinct from the offset in the file.

### 2.4 cuFile File Handle APIs

The cuFileHandleRegister API makes a file descriptor or handle that is
known to the cuFile subsystem by using an OS-agnostic interface. The API
returns an opaque handle that is owned by the cuFile subsystem.

To conserve memory, the cuFileHandleDeregister API is used to release
cuFile-related memory objects. Using only the POSIX close will not clean
up resources that were used by cuFile. Additionally, the clean up of
cufile objects that are associated with the files that were operated on
in the cuFile context will occur at cuFileDriverClose.

[CUfileError_t cuFileHandleRegister(CUFileHandle_t \*fh, CUFileDescr_t
\*descr);]{.mark}

[void cuFileHandleDeregister(CUFileHandle_t fh);]{.mark}

### 2.5 cuFile buffer APIs

The cuFileBufRegister API incurs a significant performance cost, so
registration costs should be amortized where possible. Developers must
ensure that buffers are registered up front and off the critical path.

The cuFileBufRegister API is optional. If this is not used, instead of
pinning the user's memory, cuFile-managed and internally pinned buffers
are used.

The cuFileBufDeregister API is used to optimally clean up cuFile-related
memory objects, but CUDA currently has no analog to cuFileBufDeregister.
The cleaning up of objects that are associated with the buffers that
were operated on in the cuFile context occur at cuFileDriverClose. If
explicit APIs are used, the incurred errors are reported immediately,
but if the operations of these explicit APIs are performed implicitly,
error reporting and handling is less clear.

[CUfileError_t cuFileBufRegister(const void \*devPtr_base, size_t size,
int flags);]{.mark}

[CUfileError_t cuFileBufDeregister(const void \*devPtr_base);]{.mark}

## 3 cuFile API Functional Specification

### 3.1 cuFile 驱动 API Functional Specification

### 3.2 cuFile IO API Functional Specification

### 3.3 cuFile内存管理Functional Specification

### 3.4 cuFile流API Functional Specification

The stream APIs are similar to Read and Write, but they take a stream
parameter to support asynchronous operations and execute in the CUDA
stream order.

1、[[cuFileReadAsync]{.underline}](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilereadasync)

CUfileError_t cudaFileReadAsync(CUFileHandle_t fh, void \*devPtr_base,

size_t \*size, off_t file_offset,

off_t devPtr_offset,

int \*bytes_read, [cudaStream_t]{.mark} stream);

CUfileError_t cuFileReadAsync(CUFileHandle_t fh, void \*devPtr_base,

size_t \*size, off_t file_offset,

off_t devPtr_offset,

int \*bytes_read, [CUstream]{.mark} stream);

2、[[cuFileWriteAsync]{.underline}](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilewriteasync)

CUfileError_t cudaFileWriteAsync(CUFileHandle_t fh, void \*devPtr_base,

size_t \*size, off_t file_offset,

off_t devPtr_offset,

int \*bytes_written, [cudaStream_t]{.mark} stream);

CUfileError_t cuFileWriteAsync(CUFileHandle_t fh, void \*devPtr_base,

size_t \*size, off_t file_offset,

off_t devPtr_offset,

int \*bytes_written, [CUstream_t]{.mark} stream);

## 4 cuFile Batch API Functional Specification

## 5 示例代码

见cuFile_test.cu
