# Roms版本、工具包及资料文档汇总

roms 学习中，roms官方团队并不大，有很多来自于世界各地使用者贡献的程序和学习资料可以使用，善用这些可以大大节省时间，集中注意力于科学问题上.

## Roms流行的三个不同版本

1 美国New Jersey州Rutser大学http://www.myroms.org/，这是目前使用最多和文档比较全的版本，有专门对应的文档网站https://www.myroms.org/wiki/Documentation_Portal，但是很多最近更新都是在08、09年左右，比如matlab的seagrid包，netcdf包，会有兼容问题。

2 ROMS_AGRIF project 法国的一个版本，https://www.croco-ocean.org/download/roms_agrif-project/ 在这个项目的基础上，他们开发了Coastal and Regional Ocean COmmunity model （Croco），模式架构和源代码和Rutser大学的基本一样，但是编译脚本和处理工具不同，版本更新最近2014年左右。此版本Roms的matlab工具包很全，也提供常用的数据及下载。用roms模式做多层嵌套做的挺好，建议多多采用他们的工具包。

3 UCLA的版本，官网http://research.atmos.ucla.edu/roms/Welcome.html就一个架子，没什么内容，不过网上经常可以看到他们贡献的程序包


## 常用工具包

ROMS Numerical Toolbox for MATLAB （RNT），功能比较丰富，使用的人也不少，但是作者并没有做太多的整理和文档工作。

http://oces.us/RNT/

NCL roms绘图https://www.ncl.ucar.edu/Applications/roms.shtml 和http://www.ncl.ucar.edu/Training/Workshops/UMaine/Scripts/ROMS/ 绘图脚本不多，处理工具也没有matlab丰富，但是可以发挥NCL作图漂亮的优势，开发一下画图还是不错的


## roms学习资料文档

http://www.o3d.org/ROMS-Tutorial/tutorials.html E. Di Lorenzo的视频，来自于Georgia Institute of Technology
作者的亲身视频演示，采用了RNT工具包，mov格式，清晰易懂，视频质量也不错，很好的学习资料

ftp://ftp.legos.obs-mip.fr/pub/romsagrif/DATA_ROMS/papers/draft_ROMS_Manual_K.Hedstroem.pdf ROMS_AGRIF的官方技术文档draft。
