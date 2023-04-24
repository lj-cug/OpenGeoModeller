#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks 4
#SBATCH --output test.out

## 通过 -N 指令指定节点数
## 通过 --ntasks 指定处理器需求数
## 通过 --output 指定输出文件
## 通过 --time 指定启动时间
## mpirun 运行编译好的可执行程序
mpirun --allow-run-as-root -np 4 ./test.exe
