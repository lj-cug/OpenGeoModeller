# Delft3D-FM

Delft3D Flexible Mesh Flow model (非结构网格的Delft3d)

## 源码仓库

在svn仓库：	https://svn.oss.deltares.nl/repos/delft3d/tags/delft3dfm/

## 下载

svn co https://svn.oss.deltares.nl/repos/delft3d/tags/delft3dfm/65980/

## 使用

可用于[水动力教学](https://github.com/openearth/hydrodynamics-course)和工程咨询

教学程序使用的fm版本是65957, 仓库中没有, 最接近的是65980

## 编译

- build.sh   
  Execute "./build.sh --help" to show the usage   
  Currently used as default build process: "./build.sh all --compiler intel21"   
  This will execute "src/setenv.sh" on Deltares systems. On other systems, the environment must be prepared upfront. 