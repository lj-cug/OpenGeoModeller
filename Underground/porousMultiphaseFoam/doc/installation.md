# Installation-PMF-v2303

���ȣ���װ��OpenFOAM-v2206 (������v**06�汾!!!)

Ȼ�󣬼���OpenFOAM����������

source /opt/OpenFOAM-v2206/etc/bashrc

�����"porousMultiphaseFoam"·���£�ִ�У�./Allwmake -j

PMF��̬���ӿ��ļ��洢�ڱ�׼��OpenFOAM�û�·���£�  $FOAM_USER_LIBBIN
  
��ִ�������λ�ڱ�׼��OpenFOAM�û�·���£� $FOAM_USER_APPBIN

- Each tutorial directory contains "run" and "clean" files to test installation
  and validate the solver.

- A python script runTutorials.py can be used to test all components.

- To remove compilation and temporary files, run ::

./Allwclean --purge