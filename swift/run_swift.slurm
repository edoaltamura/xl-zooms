#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --exclusive
#SBATCH -J RUN_NAME
#SBATCH -o ./logs/%x.%J.out
#SBATCH -e ./logs/%x.%J.err
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH -t 72:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load intel_comp/2018
module load intel_mpi/2018
module load parmetis/4.0.3
module load parallel_hdf5/1.8.20
module load gsl/2.4
module load fftw/3.3.7
module load python/3.6.5
module load cosma/2018
module load cmake

# Run swift
$(pwd)/../swiftsim/examples/swift \
  -v 1 \
  -e \
  --cosmology \
  --eagle \
  --pin \
  --threads=$SLURM_CPUS_PER_TASK \
  eagle_-8res.yml #> "./logs/swift.stdout" 2>"./logs/swift.stderr"

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode