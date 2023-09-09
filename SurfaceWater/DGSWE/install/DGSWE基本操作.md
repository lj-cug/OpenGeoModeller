# DGSWE��������

��run_case.py���룺

ʹ��python2���Ե�subprocess module���Ƹ���������У�

show_output��ʾִ��ĳ������̣��������Ϣ��

def show_output(command):

output = subprocess.Popen(command,stdout=subprocess.PIPE,shell=True)

while output.poll() is None:

l = output.stdout.readline()

print l.rstrip(\'\\n\')

## 1�����þ���·��

plot_work_dir

dgswe_work_dir

example_dir

nprocessors #���õĽ�����(MPI)

## 2������DG-SWEM

��1������/workĿ¼������dgswe_mpi����

make clean

make metis

make dgswe OPTS=mpi

make dgprep

make dgpost

ע�⣺���������������󣬽��������"DGSWEM����������.md"

��2������/plot/workĿ¼������plot����

make clean

make plot

��3������dgswe

-   ����һ���ļ���np.in��������nprocessors

-   ִ��ǰ����./dgprep \< np.in

-   ִ��������mpirun --n nprocessors ./dgswe_mpi

-   ��ִ�к���./dgpost

��4�����ƣ����ӻ��������

./plot

## 3��ɾ�������������Ӳ������ļ�

clean_case.py

ɾ�����е��ļ���������ִ�г���

## plot�������

���ΪPostScript��ʸ��ͼ�������߽�FEM�Ŀ��ӻ���������Brus,
2017�Ĳ�ʿ���ģ�

��Ҫ�����ļ���plot.inp��plot_sta.inp (��ѡ)

plot.inp�ļ�������Ȥ�Ĳ�����

station plot option ���Ʋ�վλ�õ�ʾ��ͼ

order of nodal set for plotting straight elements

order of nodal set for plotting curved elements

adaptive plotting option

colormap path ��ѡ��Legend��ɫģʽ

plot Google Map ���عȸ��ͼͼƬ����Ϊ����ͼ

## error��������

���������Ժ����ĺ�����򣬱Ƚϴ������ϸ������������ʡ�

��Ҫ�����ļ�error.inp

!/home/sbrus/Codes/dgswe/grids/converge_quad.grd ! coarse grid file

!/home/sbrus/data-drive/converge_quad/mesh1/P2/CTP2/ ! coarse output
directory

!2 ! p - coarse polynomial order

!2 ! ctp - coarse parametric coordinate transformation order

!.5d0 ! dt - coarse timestep

!/home/sbrus/Codes/dgswe/grids/converge_quad2.grd ! fine grid file

!/home/sbrus/data-drive/converge_quad/mesh2/P2/CTP2/ ! fine output
directory

!2 ! p - fine polynomial order

!2 ! ctp - fine parametric coordinate transformation order

!.25d0 ! dt - fine timestep

!2d0 ! tf - final time (days)

!20 ! lines - lines in output files

## bathy_interp����

���εĸ߽ײ�ֵ���㡣

�����ļ���bathy.inp�����ݣ�

����ļ���\_interp.hb��elem_nodes.d��interp_nodes.d��boundary_nodes.d��bathy.d

## rimls����

�����ļ���rimls.inp

���幦����δ��������Ż�����ģ�

/home/sbrus/Codes/dgswe/grids/dummy.cb ! curved boundary file

/home/sbrus/Codes/dgswe/rimls/work/Rimls_test-sub.grd ! eval grid - used
to determine rimls surface evaluation points

1 ! eval hbp - bathymetry order

1 ! eval ctp - parametric coordinate transformation order

/home/sbrus/Codes/dgswe/grids/dummy.cb ! curved boundary file

3 ! lsp - moving least squares fit order

0 ! basis_opt - basis for least squares polynomial (1 - orthonormal,
else - simple)

1d0 ! Erad - radius of Earth

0d0,0d0 ! lambda0,phi0 - center of CPP coordinate system

3.0d0 ! r - muliplier for search radius (1.5 - 4.0)

1.5d0 ! sigma_n - smoothing parameter (0.5 - 1.5)

../output/ ! output directory

1 ! nrpt - number of random points (for converging channel hardwire)

0d0

## spline����

�����ļ���spline.inp

## stations����

�����ļ���dgswe.inp

���Ʋ�վ��λ��ʾ��ͼ��

## util�ļ���

���ļ����°����ܶ๤��С����FORTRAN��MATLAB���ԡ�