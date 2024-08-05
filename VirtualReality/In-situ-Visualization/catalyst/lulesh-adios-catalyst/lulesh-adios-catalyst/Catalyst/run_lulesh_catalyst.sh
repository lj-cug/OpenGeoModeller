#!/bin/bash -l
#SBATCH --job-name="lulesh+catalyst"
#SBATCH --account="csstaff"
##SBATCH --reservation="in-situ"
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

module load daint-gpu ParaView

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VTK_SILENCE_GET_VOID_POINTER_WARNINGS=1

cp catalyst.py ../buildCatalyst/bin/lulesh2.0 $SCRATCH
pushd $SCRATCH
srun ./lulesh2.0 -x catalyst.py -s 30 -p

