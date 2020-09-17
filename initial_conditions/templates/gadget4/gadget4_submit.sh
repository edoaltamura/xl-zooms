#!/bin/bash

#SBATCH --output=log.txt
#SBATCH --time=24:00:00
#SBATCH --partition=cosma7
#SBATCH --account=dp004
#SBATCH --exclusive
#SBATCH --ntasks=28
#SBATCH --cpus-per-task=1

module purge
module load intel_comp/2018
module load intel_mpi/2018
module load hdf5/1.10.3
module load gsl
module load fftw/3.3.7
echo "Loaded gadget 4 modules (serial HDF5)."

mpirun -np $SLURM_NTASKS /cosma7/data/dp004/rttw52/codes/gadget4/Gadget4 params.txt

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit

