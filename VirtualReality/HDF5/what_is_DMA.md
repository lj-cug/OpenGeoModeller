# 1. Where Physical Address came from ?
As you can see below, /proc/iomem says not only the physical address originated from DIMM but the physical address from PCI device or BIOS. When kernel is booting, the device provides its "BAR" which the kernel then maps into main memory. The memory mappings is exposed to userspace via /proc/iomem.
```
$ sudo cat /proc/iomem 
00000000-00000fff : Reserved
00001000-0009fbff : System RAM
0009fc00-0009ffff : Reserved
000a0000-000bffff : PCI Bus 0000:00
000c0000-000c99ff : Video ROM
000ca000-000cadff : Adapter ROM
000cb000-000cd3ff : Adapter ROM
000f0000-000fffff : Reserved
  000f0000-000fffff : System ROM
00100000-7ffcffff : System RAM
7ffd0000-7fffffff : Reserved
80000000-afffffff : PCI Bus 0000:00
b0000000-bfffffff : PCI MMCONFIG 0000 [bus 00-ff]
  b0000000-bfffffff : Reserved
c0000000-febfffff : PCI Bus 0000:00
  d0000000-efffffff : PCI Bus 0000:04
    d0000000-dfffffff : 0000:04:00.0
    e0000000-e1ffffff : 0000:04:00.0
  f0000000-f07fffff : 0000:00:01.0

... snip ...

  fb000000-fcffffff : PCI Bus 0000:04
    fb000000-fbffffff : 0000:04:00.0
      fb000000-fbffffff : nvidia
    fc000000-fc07ffff : 0000:04:00.0
  fd000000-fd1fffff : PCI Bus 0000:09
  fd200000-fd3fffff : PCI Bus 0000:08
  fd400000-fd5fffff : PCI Bus 0000:07
  fd600000-fd7fffff : PCI Bus 0000:06
    fd600000-fd603fff : 0000:06:00.0
      fd600000-fd603fff : nvme
    
... snip ...

feffc000-feffffff : Reserved
fffc0000-ffffffff : Reserved
100000000-27fffffff : System RAM
  135000000-135e00eb0 : Kernel code
  135e00eb1-13685763f : Kernel data
  136b20000-136ffffff : Kernel bss
280000000-a7fffffff : PCI Bus 0000:00
```
# 2. PCI BAR space
BAR is Base Address Register, But I believe that Not all of engineers can understand what it is from the name itself. BAR register describes a region that is between 16 bytes and 2 gigabytes in size, located below 4 gigabyte address space limit. If a platform supports the "Above 4G" option in system firmware, 64 bit bars can be used. The following is a result of my GPU's lspci which says there are the three regions allocated for the purpose of DMA or MMIO. Yes, one of these regions is for DMA between host memory and device's memory used by the controller's logic behind its memory.
(ex) GPU's memory is for the calculation on GPU card or NIC's memory is for the sending/receiving messages thru the port.
```
$ sudo lspci -v -x -s 04:00.0
04:00.0 VGA compatible controller: NVIDIA Corporation GP107GL [Quadro P1000] (rev a1) (prog-if 00 [VGA controller])
	Subsystem: NVIDIA Corporation GP107GL [Quadro P1000]
	Physical Slot: 0-3
	Flags: bus master, fast devsel, latency 0, IRQ 48
	Memory at fb000000 (32-bit, non-prefetchable) [size=16M]
	Memory at d0000000 (64-bit, prefetchable) [size=256M]
	Memory at e0000000 (64-bit, prefetchable) [size=32M]
	I/O ports at c000 [size=128]
	Expansion ROM at fc000000 [virtual] [disabled] [size=512K]
	Capabilities: [60] Power Management version 3
	Capabilities: [68] MSI: Enable+ Count=1/1 Maskable- 64bit+
	Capabilities: [78] Express Legacy Endpoint, MSI 00
	Capabilities: [100] Virtual Channel
	Capabilities: [250] Latency Tolerance Reporting
	Capabilities: [128] Power Budgeting <?>
	Capabilities: [420] Advanced Error Reporting
	Capabilities: [600] Vendor Specific Information: ID=0001 Rev=1 Len=024 <?>
	Kernel driver in use: nvidia
	Kernel modules: nvidiafb, nouveau, nvidia_drm, nvidia
00: de 10 b1 1c 07 05 10 00 a1 00 00 03 00 00 00 00
10: 00 00 00 fb 0c 00 00 d0 00 00 00 00 0c 00 00 e0
20: 00 00 00 00 01 c0 00 00 00 00 00 00 de 10 bc 11
30: 00 00 00 00 60 00 00 00 00 00 00 00 0b 01 00 00
```

See again the /proc/iomem and compare it with the lspci.
```
    ...
  d0000000-efffffff : PCI Bus 0000:04
    d0000000-dfffffff : 0000:04:00.0   ---> Memory at d0000000 (64-bit, prefetchable) [size=256M]
    e0000000-e1ffffff : 0000:04:00.0   ---> Memory at e0000000 (64-bit, prefetchable) [size=32M]
    ...
    fb000000-fcffffff : PCI Bus 0000:04
    fb000000-fbffffff : 0000:04:00.0   ---> Memory at fb000000 (32-bit, non-prefetchable) [size=16M]
      fb000000-fbffffff : nvidia
    fc000000-fc07ffff : 0000:04:00.0   ---> Expansion ROM at fc000000 [virtual] [disabled] [size=512K]
```

You also compare it with the files in /sys/bus/pci/devices. See below:
```
$ cd /sys/bus/pci/devices/0000:04:00.0
$ ls -l resource*
-r--r--r-- 1 root root      4096  8月  1 18:43 resource
-rw------- 1 root root  16777216  8月  1 19:29 resource0     ---> Memory at fb000000 (32-bit, non-prefetchable) [size=16M]
-rw------- 1 root root 268435456  8月  1 19:29 resource1     ---> Memory at d0000000 (64-bit, prefetchable) [size=256M]
-rw------- 1 root root 268435456  8月  1 19:29 resource1_wc
-rw------- 1 root root  33554432  8月  1 19:29 resource3     ---> Memory at e0000000 (64-bit, prefetchable) [size=32M]
-rw------- 1 root root  33554432  8月  1 19:29 resource3_wc
-rw------- 1 root root       128  8月  1 19:29 resource5     ---> I/O ports at c000 [size=128]
```
# 3. What is DMA ?
DMA is a copy of data between the PCI device's memory and host memory without CPU load. 
DMA is very similar to MMIO's behavior but DMA is performed by DMA engine on the PCI device not by CPU. 
DMA engine should fetch some instructions created by the Device Driver from host memory in advance.
The instruction has the kernel's pysical memory address which the DMA engine can copy the data from the BAR space to.
After interrups from DMA engine which already finished the copy, CPU will copy it to the User space via virtual address. Of cource it needs address translations and so heavy overheads.
Especially, if the address in the instruction refers user space, then we call it RDMA or Kernel Bypass. But then, please note the translation between physical address to virtual address should be performed on the PCI device's DMA engine itself not by OS and CPUs!
```
                    Physical Memory
                    +----------+
                    |          |
                    |          |
                    +----------+ 0xdfffffff
      +------------ |XXXXXXXXXX|
      |      +----- |XXXXXXXXXX|
      |      |      +----------+ 0xd0000000 (GPU BAR#1)
      |      |      |          |                                          Kernel Space (Virtual Address)
      |    Copy     |          |                                          +----------+
      |    (DMA)    |          |                                          |          |
      |      |      +----------+                                          |          | 
    Kernel   +----> |XXXXXXXXXX|                                          |XXXXXXXXXX|
    Bypass   +----- |XXXXXXXXXX| <================Mapping===============> |XXXXXXXXXX| -----+
      |      |      +----------+ Host Memory for DMA operation            +----------+      |
      |    Copy     |          | (Physical Address)                                        Copy
      |    (CPU)    |          |                                          +----------+     (CPU)
      |      |      +----------+                                          |          |      |
      |      +----> |XXXXXXXXXX|                                          |XXXXXXXXXX| <----+
      +-----------> |XXXXXXXXXX| <================Mapping===============> |XXXXXXXXXX|
	            +----------+ User Space (Physical Address)            +----------+
                    |          |                                          User Space (Virtual Address)
                    |          |
                    +----------+ 0x00000000
```
See also below: 

https://stackoverflow.com/questions/3851677/what-is-the-difference-between-dma-and-memory-mapped-io
https://softwareengineering.stackexchange.com/questions/272470/how-does-a-dma-controller-work
