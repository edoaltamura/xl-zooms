#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --exclusive
#SBATCH -J EAGLE-XL_ClusterSK0_HYDRO
#SBATCH -o ./logs/%x.%J.out
#SBATCH -e ./logs/%x.%J.err
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH -t 10:00:00
#

source ../modules.sh

# Stripe snapshots only
lfs setstripe --stripe-count=1 --stripe-size=16M ./restart/
lfs setstripe --stripe-count=2 --stripe-size=16M ./snapshots/

# Start from ICs if we have a file 'begin'
#restartflag="--restart"
#if [ -e begin ] ; then
#  \rm -f begin
#  restartflag=""
#fi

# Use ${restartflag}

# Run swift
mpirun -np $SLURM_NTASKS \
    ./swiftsim-hydro/examples/swift \
    --verbose=1 --cosmology --eagle --pin --velociraptor\
    --threads=$SLURM_CPUS_PER_TASK \
    ./parameters_hydro.yml 2>&1 | tee ./logs/stdout.log

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode
exit