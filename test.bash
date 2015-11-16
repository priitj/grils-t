#!/bin/bash

#SBATCH --job-name=gop-grasp-test
#SBATCH --output=gop-grasp-test-%J.log
#SBATCH --ntasks=2

PROBLIST=$1
[ "X" = "X${PROBLIST}" ] && exit 1

# TUT HPC cluster quirk
source /etc/profile.d/modules.sh

# for the mpi4py module
module load python-2.7.5

#mpiexec -n 2 python ./op.py gop/op3c.gop
time ./runtests.sh ../optests1/traj_elim1_d10 op ${PROBLIST} 2 3 100
time ./runtests.sh ../optests1/pr_2opt op_pr ${PROBLIST} 2 3 100

