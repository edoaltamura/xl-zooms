#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --exclusive
#SBATCH -J EAGLE-XL_ClusterSK0_HYDRO
#SBATCH -o ./logs/%x.%J.out
#SBATCH -e ./logs/%x.%J.err
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH -t 60:00:00

export OMP_NUM_THREADS=$SLURM_NTASKS

module purge
module load intel_comp/2020
module load intel_mpi/2020
module load parmetis/4.0.3
module load parallel_hdf5/1.10.3
module load gsl/2.4
module load fftw/3.3.7
module load cmake
module load python/3.6.5
module load ffmpeg/4.0.2

# Start from ICs if we have a file 'begin'
restartflag="--restart"
if [ -e begin ] ; then
  \rm -f begin
  restartflag=""
fi

# Use

# Run swift
./swiftsim-hydro/examples/swift \
  ${restartflag} \
  -v 1 \
  --cosmology \
  --eagle \
  --pin \
  --threads=$OMP_NUM_THREADS \
  ./config/parameters_hydro.yml 2>&1 | tee ./logs/stdout.log

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode
exit