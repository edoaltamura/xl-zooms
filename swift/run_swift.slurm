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

source modules.sh

# Run swift
../swift \
  -v 1 \
  --cosmology \
  --eagle \
  --pin \
  --threads=$OMP_NUM_THREADS \
  eagle_-8res.yml 2>&1 | tee ./logs/stdout_swift.log

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

sbatch ./run_vr.slurm