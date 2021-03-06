#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --exclusive
#SBATCH -J RUN_NAME
#SBATCH -o ./logs/%x.%J.swift.out
#SBATCH -e ./logs/%x.%J.swift.err
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH -t 72:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load intel_comp/2018
module load intel_mpi/2018
module load parmetis/4.0.3
module load parallel_hdf5/1.10.3
module load gsl/2.4
module load fftw/3.3.7
module load python/3.6.5
module load cosma/2018
module load cmake

# Run swift
/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/swiftsim/examples/swift \
  -v 1 \
  --cosmology \
  --eagle \
  --pin \
  --threads=$SLURM_CPUS_PER_TASK \
  ./config/PARAM_FILE.yml

echo "Submitting VR job..."
sbatch ./run_vr.slurm

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode