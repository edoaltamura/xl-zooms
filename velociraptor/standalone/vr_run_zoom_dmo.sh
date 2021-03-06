#!/bin/bash -l

# Bash script that runs VELOCIRAPTOR-STF on a given snapshot.

source ./modules.sh

export OMP_NUM_THREADS=24

author="SK"

for i in 0 1 2
do

  out_name="halo_${author}${i}_0001"
  config_file="config_zoom_dmo.cfg"
  stdout_name="vr_output_${out_name}.stdout"
  stderr_name="vr_output_${out_name}.stderr"

  # Specify the path to the snapshot (without the final .hdf5 extension)
  snap_path="/cosma6/data/dp004/rttw52/EAGLE-XL/EAGLE-XL_Cluster${author}${i}_DMO/snapshots/EAGLE-XL_Cluster${author}${i}_DMO_0001"

  outpath="/cosma6/data/dp004/dc-alta2/xl-zooms/${out_name}"
  stdout_path="${outpath}/${stdout_name}"
  stderr_path="${outpath}/${stderr_name}"

  echo "Running VR for ${out_name}"
  if [ -d $outpath ]; then
    rm -rf $outpath
  fi
  mkdir $outpath

  # Change the config file for the output number
  cp ./$config_file $outpath
  sed 's/SNAP/0001/' $outpath/$config_file

  ./stf -i $snap_path -I 2 -o $outpath -C $outpath/$config_file > $stdout_path 2>$stderr_path

  # Move outputs inside directory
  mv /cosma6/data/dp004/dc-alta2/xl-zooms/$out_name.* $outpath

done

