# ԭλ���ӻ�
Catalyst��ʾ�����뼰ԭ����μ�

# Ӧ�ó���
`
RegESM��쫷���ƶ����̣�����paraview-catalyst
BloodFlow�����Ӷ���ѧLAMMPS + LBMģ����Palabos�� + SENSEI���߿��ӻ�ϵͳ��ʵ��Ѫ���к�ϸ���˶���ģ�⼰���ӻ�
Nek5000 (2021, KTH Sweden)
CESM_1.0
miniFE
`

# �ο�����
libMesh-sedimentation�� Jose J. Camata, et al. In situ visualization and data analysis for turbidity currents simulation. Computers and Geosciences 110 (2018) 23�C31

# �ܽ�
���ģ��ѧ�����IO��Ϊƿ��������£����߿��ӻ���Ϊ��ǰ���о��ȵ㡣
��ͬ��Ӳ���ܹ��Ĳ��м��㣬�����߿��ӻ�Ҳ������µ�Ҫ��
ParaView Catalyst��VisIT Libsim�����߿��ӻ������������кܶ�CFDӦ��ʹ����Catalyst����Nek500, RegESM, PyFR, CAM, ...
��ͳ�����߿��ӻ����ǽ����ݴ�CPU���ڴ棩ת�Ƶ�GPU�ڴ棬��Catalyst
Խ��Խ��Ĵ���Ǩ�Ƶ�GPU��CUDA����������ʵ����GPU��ʵ�ּ�������ӻ����νӣ��Ǻܾ�����ս���о�����ġ�
VTK-m��Ϊ�ν�GPU��������ӻ����м���������չ������о�����PyFR-Catalyst

�����˸�ͳһ�����߿��ӻ���ܣ�����SENSEI, Damaris��Ascent��
SENSEI��Ascent��������DOE��LLNL���������߿��ӻ�ͳһ��ܣ�Damaris�Ƿ���inria���������߿��ӻ�ͳһ��ܡ�����ͨ��XML�����ļ��͵���API��ʵ��MPI+���/�ں˵����ݿ��ӻ���

## Catalyst and Libsim
ParaView��VISIT�����ԭ��ԭλ���ӻ���

## Ascent (�Ƽ�)
Ascent�з��ĳ��򣬿�֧��MPI+OpenMP/CUDA�����߿��ӻ���ͨ��VTK-m�м�㡣
��װ��Catalyst and Libsim
Ascent֧��C++��Python����
�кܶ�Mini-app: LULESH, Coverleaf3D��

## SENSEI
SENSEI������Catalyst, Ascent��ADIOS2(����Ӧ���ݹ�����)

## Damaris
Damaris��һ�������������߿��ӻ�ͳһ��ܣ����Զ�FORTRAN/C/C++��ģ����룬����ʵʩVisIT��ParaView�����߿��ӻ���
Damaris-1.5.0��ʼ֧��ParaView�µķǽṹ���������ݵ����߿��ӻ���û��1.4�汾�������ݲ�֧��VisIT�ķǽṹ�������ݵĿ��ӻ���
û�й�����tutorials !

## ADIOS2
In transit analysis:  ADIOS2-Catalyst
