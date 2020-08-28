#!/bin/bash -l

#SBATCH --ntasks=112
#SBATCH -J MakeParticleLoad
#SBATCH --output=out_files/submit_out.log
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1

python_mpi

mpirun -np $SLURM_NTASKS python3 Generate_PL.py pl_param_files/TABULA/L1200/150Mpc_Slab_512.yml
mpirun -np $SLURM_NTASKS python3 Generate_PL.py pl_param_files/TABULA/L1200/150Mpc_Slab_256.yml
mpirun -np $SLURM_NTASKS python3 Generate_PL.py pl_param_files/TABULA/L1200/150Mpc_Slab_128.yml

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode
exit

