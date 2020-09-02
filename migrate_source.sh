#!/bin/bash
destination_directory=/cosma7/data/dp004/dc-alta2/xl-zooms
subdirectory=HYDRO
run_name=EAGLE-XL_ClusterSK0

mkdir -p destination_directory/$subdirectory
mkdir -p destination_directory/$subdirectory/$run_name
mkdir -p destination_directory/$subdirectory/$run_name/logs

cp ./swift/parameters_hydro.yml destination_directory/$subdirectory/$run_name
cp ./swift/run_hydro.slurm destination_directory/$subdirectory/$run_name
cp ./swift/snap_redshifts.txt destination_directory/$subdirectory/$run_name

sbatch destination_directory/$subdirectory/$run_name/run_hydro.slurm
