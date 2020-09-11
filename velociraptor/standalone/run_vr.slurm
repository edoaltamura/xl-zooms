#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --exclusive
#SBATCH -J VR-RUN_NAME
#SBATCH -o ./logs/%x.%J.vr.out
#SBATCH -e ./logs/%x.%J.vr.err
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
module load python/3.6.5
module load cosma/2018
module load cmake

function get_snap_index
{
  filename=$1
  tmp=${filename#*_}
  num=${tmp%.*}
  echo "$num"
}

mkdir -p "$PWD/stf"

# Make a dmo/hydro switch
if [[ $PWD == *"hydro"* ]]; then
  switch_mode="hydro"                 # If this file in "/hydro" subdirectory, use hydro VR and hydro parameter file
else
  switch_mode="dmo"                   # If this file in "/dmo" subdirectory, use dmo VR and hydro parameter file
fi

config_template="./config/vr_config_zoom_$switch_mode.cfg"

for snap_path in ./snapshots/*.hdf5; do

  snap_name=$(basename "$snap_path")                                      # Delete the path info in $snap_path and retain "filename.hdf5"
  base_name=$(echo "$snap_name" | cut -f 1 -d '.')                        # Delete the file extension in $name and retain "filename"
  inpath="$PWD/snapshots/$base_name"                                      # Specify the path of the input snapshot
  outpath="$PWD/stf/$base_name"                                           # Specify the path of the output catalogs
  mkdir -p "$outpath"                                                     # Create a directory named as "filename"
  cp "$config_template" "$outpath"                                        # Copy template config file in output directory
  mv "$outpath/$(basename $config_template)" "$outpath/$base_name.cfg"    # Rename with its corresponding snapshot name
  sed -i "s/SNAP/$(get_snap_index $snap_name)/" "$outpath/$base_name.cfg" # Replace snapshot-specific data in cfg template
  stdout_path="$PWD/stf/$base_name/$base_name.stdout"                     # Create file paths for stdout in output directory
  stderr_path="$PWD/stf/$base_name/$base_name.stderr"                     # Create file paths for stderr in output directory

  /cosma7/data/dp004/dc-alta2/xl-zooms/$switch_mode/VELOCIraptor-STF/stf \
   -i "$inpath" \                                                         # Input snapshot (without file extension)
   -I 2 \                                                                 # Specify input extension (2 = .hdf5)
   -o "$outpath" \                                                        # Path for output catalogs (saved in same directory as $outpath, not inside it)
   -C "$outpath/$base_name.cfg" > "$stdout_path" 2>"$stderr_path"         # Parameter file and stdout/stderr log files

  mv "$PWD/stf/$base_name".* "$outpath"                                   # Move outputs of VR inside $outpath directory

done

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode