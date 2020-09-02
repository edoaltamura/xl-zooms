#!/bin/bash
destination_directory=/cosma7/data/dp004/dc-alta2/xl-zooms/hydro
run_name=EAGLE-XL_ClusterSK0

mkdir -p /cosma7/data/dp004/dc-alta2/xl-zooms
mkdir -p $destination_directory
mkdir -p $destination_directory/$run_name
mkdir -p $destination_directory/$run_name/logs

cp ./swift/parameters_hydro.yml $destination_directory/$run_name
cp ./swift/run_hydro.slurm $destination_directory/$run_name
cp ./swift/snap_redshifts.txt $destination_directory/$run_name

cd $destination_directory/$run_name
sbatch ./run_hydro.slurm
squeue -u dc-alta2
