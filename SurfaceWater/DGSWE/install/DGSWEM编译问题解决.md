# ����dgswem���ֵĴ��󼰽������

\(1\)

Error: There is no specific subroutine for the generic
\'mpi_dist_graph_creat_adjacent\'

**�������**��ʹ��mpich2��(mpich-3.3)������ʹ��OpenMPI(5.0�汾����)��

gedit \~/.bashrc

export PATH=/home/lijian/mpich-3.3/bin:\$PATH

\(2\) ��\'dgesv\_\"δ�����Ӧ��

û����ȷ��װ������lapack�⡣

��/work/makefile���Ķ���118����������ݣ�

\###############################################################

\# Library Links (Platform Specific) \#

\###############################################################

�鿴��������hostname

�޸�makefile�е�ifeq
(\$(NAME),chl-tilos)Ϊ��lijian���ҵıʼǱ�����������

\(3\) ʹ��Python 3ִ��run_case.py���֣�print l.strip()���﷨����

[�л�ΪPython 2.7]{.mark}

alias python=\'/usr/bin/python2.7\'

��bashrc��ʹ�ü�������������Ч��

���߽��������ӣ�ln --s /usr/bin/python2.7 /usr/bin/python

����ʹ��update-alternatives

\(4\) ִ��./plot���ִ���convert not authorized error/..

�޸ģ�/etc/ImageMagick-6/policy.xml�ļ��е����ݣ�

rights=\"none\" rights=\"read\|write\"

# python�Ľ��̹����﷨

os.chdir()�������ڸı䵱ǰ����Ŀ¼��ָ����·����

os.rename()�������������ļ���Ŀ¼����src��dst�����dst��һ�����ڵ�Ŀ¼�����׳�OSError��

os.rename�﷨

rename()�����﷨��ʽ���£�

os.rename(src, dst)

����

src \-- Ҫ�޸ĵ�Ŀ¼��

dst \-- �޸ĺ��Ŀ¼��