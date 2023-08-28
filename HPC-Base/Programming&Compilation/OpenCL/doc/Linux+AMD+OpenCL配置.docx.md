**安装环境**

换了AMD的显卡：以后都改用AMD的卡了

**一、下载对应驱动并安装**

https://support.amd.com/zh-cn/download  
结果我选好了型号没反应？不要这样，直接在这个网页的下部分：最新AMD驱动程序
 那里找。我找到：https://support.amd.com/zh-cn/download/linux
 我的是CentOS，所以选择：https://support.amd.com/en-us/kb-articles/Pages/AMDGPU-PRO-Driver-for-Linux-Release-Notes.aspx
 我选的是amdgpu-pro-17.30-465504.tar.xz 这个驱动下载完毕  
准备按照https://support.amd.com/en-us/kb-articles/Pages/AMDGPU-PRO-CentOS-Install-Uninstall.aspx
 来安装
：在开始"Pre-Install"这一步之前，我先下载 amdgpu-pro-preinstall.sh
放在amdgpu-pro-17.30-465504文件夹下，然后运行：sh
amdgpu-pro-preinstall.sh
 。然后再按照这个网址后部分的进行，即"Install"即可，时间略长，耐心等待。这一步完毕重启后发现字体明显变小了，这是驱动安装成功的表现。检测一下到底有没有安装成功，运行：rpm
-qa \| grep amdgpu-pro  即可出现类似我电脑的提示

二、下载APP-SDK

在http://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/
下下载APPSDK而我是 在

<https://pan.baidu.com/share/link?shareid=463196839&uk=1094854304&errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0>

然后按照这个压缩文件解压后的文件夹下有一个readme。按照readme的第1步执行即可。第一步执行完毕后，可以在opt目录下看到AMDAPP文件夹，即成功。
readme上其它十几个问答可以稍微看看即可。后面检测到这个SDK太老runtime太老与我的驱动冲突
 所以监测不到我的GPU   无奈之下
我又卸载这个版本的APPSDK(为了彻底卸载干净又必须要卸载驱动amdgpu-pro-uninstall)
又重新下新的.所以一定不要像我一样下一个旧的APP-SDK 害死人啊  
 ！！！下一个3.0的最好。

终于检测到了GPU：

**三、下载CodeXL并安装**

<https://github.com/GPUOpen-Tools/CodeXL/releases>

我下载的是：[CodeXL_Linux_x86_64_2.4.60.tar.gz](https://github.com/GPUOpen-Tools/CodeXL/releases/download/v2.4/CodeXL_Linux_x86_64_2.4.60.tar.gz) 
然后解压进入目录，运行：./CodeXL 即可出现画面：

再点击help下的load the Teapot sample 等待即可出现：

成功。

**二、创建并运行工程**

以前是在Nvidia下写的工程 ，现在要在AMD下：
肯定把include和lib换过来,还有换成#include\<CL/cl.h\> 而不再是hpp

运行成功！！！

但运行另一个工程时失败，原因是：以前N卡下clCreateBuffer()的第4个参数我是有变量的
  而A卡下第4个参数只能是NULL或者0！！！

这个卡果然快了很多啊！
