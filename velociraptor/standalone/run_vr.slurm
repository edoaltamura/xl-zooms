#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --exclusive
#SBATCH -J VR-EAGLE-XL_ClusterSK0_HYDRO
#SBATCH -o ./logs/%x.%J.out
#SBATCH -e ./logs/%x.%J.err
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH -t 60:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load intel_comp/2018
module load intel_mpi/2018
module load parmetis/4.0.3
module load parallel_hdf5/1.10.3
module load gsl/2.4
module load fftw/3.3.7
module load cmake
module load python/3.6.5
module load ffmpeg/4.0.2
module load llvm/7.0.0

config_template=./config/vrconfig_3dfof_subhalos_SO_hydro.cfg

function get_snap_index
{
  filename=$1
  tmp=${filename#*_}
  num=${tmp%.*}
  echo "$num"
}

mkdir -p "$PWD/stf"

for snap_path in ./snapshots/*.hdf5; do

  # NOTE: $snap_path contains also the relative path to the hdf5 file
  snap_name=$(basename "$snap_path")                                    # Delete the path info in $snap_path and retain "filename.hdf5"
  base_name=$(echo "$snap_name" | cut -f 1 -d '.')                      # Delete the file extension in $name and retain "filename"
  out_dir_name=$base_name                                               # Make it the name of output directory
  outpath="$PWD/stf/$out_dir_name"                                      # Create a directory named as "filename"
  mkdir -p "$outpath"

  cp $config_template "$outpath"                                        # Copy template config file in output directory
  mv "$outpath/$(basename $config_template)" "$outpath/$base_name.cfg"  # Rename with its corresponding snapshot name
  config_file="$outpath/$base_name.cfg"                                 # Create new file path
  sed -i "s/SNAP/$(get_snap_index $snap_name)/" "$config_file"          # Replace snapshot-specific data in cfg template
  stdout_path="$PWD/stf/$out_dir_name/$base_name.stdout"                # Create file paths for stdout in output directory
  stderr_path="$PWD/stf/$out_dir_name/$base_name.stderr"                # Create file paths for stderr in output directory

  ./VELOCIraptor-STF-hydro/stf -i "$snap_path" -I 2 -o "$outpath" -C "$config_file" > "$stdout_path" 2>"$stderr_path"

  mv "$PWD/stf/$base_name".* "$outpath"                                 # Move outputs inside directory

done

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode