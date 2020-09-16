#!/bin/bash -l

#SBATCH -n XXX
#SBATCH -J XXX
#SBATCH -o out_files/XXX.dump
#SBATCH -e out_files/XXX.err
#SBATCH -p cosma6
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 730

module purge
module load intel_comp/2018 intel_mpi/2018 fftw/2.1.5 hdf5/1.8.20 gsl/2.4

# Workaround for the message size
export I_MPI_DAPL_CHECK_MAX_RDMA_SIZE=enable
export I_MPI_DAPL_MAX_MSG_SIZE=1073741824

# Pin the processes
export I_MPI_PIN=on
export I_MPI_PIN_MODE=pm

unset I_MPI_HYDRA_BOOTSTRAP
export I_MPI_ADJUST_ALLGATHER=1

mpirun -np $SLURM_NTASKS ../../codes/Eagle2/XXX params.param 1

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
