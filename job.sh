#!/bin/bash
#SBATCH --job-name="AC"
#SBATCH --time=1:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --constraint=rome
#SBATCH --partition=ccq
#SBATCH --exclusive
#SBATCH --output=AC_out.%j
#SBATCH --error=AC_err.%j

#======START=====

# set OMP_NUM_THREADS so that times ntasks-per-node is the total number of cores on each node
export OMP_NUM_THREADS=1

#module purge
module load slurm
module load triqs/3_unst_llvm_ompi

# with map by socket a maximum of number of cores per physical cores are spawned! This is cores per node/2
# if more threads are needed switch socket -> node
mpirun --map-by node:pe=$OMP_NUM_THREADS python3 mag_AC_2.py

#=====END====





