#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks 4
#SBATCH --output test.out

## ͨ�� -N ָ��ָ���ڵ���
## ͨ�� --ntasks ָ��������������
## ͨ�� --output ָ������ļ�
## ͨ�� --time ָ������ʱ��
## mpirun ���б���õĿ�ִ�г���
mpirun --allow-run-as-root -np 4 ./test.exe
