CMAKE 编译流程大致是：

（1）下载CMAKE:  https://cmake.org/  

  下载Windows版本的，下载高版本的Download Latest Release。

（2）在SWMM程序包力新建文件夹build

（3）打开CMAKE。

 Where is the soure code :填写到SWMM的code路径（包含Cmakelist.txt文件的）；

 Where to build the binaries: 填写到刚才的Build路径
 
（4）点击下面的Configure，选择默认的编译器（VS2013，可选64位编译）；
 
（5）检查完成后，没问题，再点击Generate。会在Build文件夹下面生成类似swmm.sln的VS工程项目。
 
（6）用Visual Studio 2013打开，就可以编译了。
 