#!/bin/bash -l

#SBATCH -n XXX
#SBATCH -J XXX
#SBATCH -o out_files/XXX.dump
#SBATCH -e out_files/XXX.err
#SBATCH -p cosma6
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 730

load_eagle

# Workaround for the message size
export I_MPI_DAPL_CHECK_MAX_RDMA_SIZE=enable
export I_MPI_DAPL_MAX_MSG_SIZE=1073741824

# Pin the processes
export I_MPI_PIN=on
export I_MPI_PIN_MODE=pm

unset I_MPI_HYDRA_BOOTSTRAP
export I_MPI_ADJUST_ALLGATHER=1

mpirun -np $SLURM_NTASKS ../../codes/Eagle2/XXX params.param 

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
