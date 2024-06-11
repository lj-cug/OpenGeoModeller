# Installation-PMF-v2303

首先，安装好OpenFOAM-v2206 (必须是v**06版本!!!)

然后，加载OpenFOAM环境变量：

source /opt/OpenFOAM-v2206/etc/bashrc

最后，在"porousMultiphaseFoam"路径下，执行：./Allwmake -j

PMF动态链接库文件存储在标准的OpenFOAM用户路径下：  $FOAM_USER_LIBBIN
  
可执行求解器位于标准的OpenFOAM用户路径下： $FOAM_USER_APPBIN

- Each tutorial directory contains "run" and "clean" files to test installation
  and validate the solver.

- A python script runTutorials.py can be used to test all components.

- To remove compilation and temporary files, run ::

./Allwclean --purge