#!/bin/bash -l

# Bash script that runs VELOCIRAPTOR-STF on a given snapshot.

source ./modules.sh

export OMP_NUM_THREADS=16

out_name="halo_SK0_0001_z000p000"
vr_loc="./stf"
config_file="config_zoom_dmo.cfg"
stdout_name="vr_output_${out_name}.stdout"
stderr_name="vr_output_${out_name}.stderr"

# Specify the path to the snapshot (without the final .hdf5 extension)
snap_path="/cosma6/data/dp004/rttw52/EAGLE-XL/EAGLE-XL_ClusterSK0_DMO/snapshots/EAGLE-XL_ClusterSK0_DMO_0001"

outpath="/cosma6/data/dp004/dc-alta2/xl-zooms/${out_name}"
stdout_path="${outpath}/${stdout_name}"
stderr_path="${outpath}/${stderr_name}"

echo "Running VR for ${out_name}"

mkdir $outpath

# Change the config file for the output number
cp ./$config_file $outpath
sed 's/SNAP/0001/' $outpath/$config_file

$vr_loc -i $snap_path -I 2 -o $outpath -C $outpath/$config_file > $stdout_path 2>$stderr_path