#!/bin/bash

echo "Initialising analysis pipeline for EAGLE-XL zoom simulations."

export OMP_NUM_THREADS=20
export NUMBA_NUM_THREADS=$OMP_NUM_THREADS
input_directory="/cosma7/data/dp004/dc-alta2/xl-zooms"
output_directory="/cosma7/data/dp004/dc-alta2/xl-zooms/analysis"

echo "OpenMP running on $OMP_NUM_THREADS threads."
echo "Python-Numba running on $NUMBA_NUM_THREADS threads."
echo "Output files will be saved in $output_directory"


# Choose which cluster and snapshot to run on
simtype="hydro"
haloid=0
resolution="-8"
snap_num=36

run_name=EAGLE-XL_ClusterSK$haloid_$resolutionres
run_directory=$input_directory/$simtype/$run_name
snapshot_name=snapshots/$run_name_00$snap_num.hdf5

python3 performance/number_of_steps_simulation_time.py \
  $run_name \
  $run_directory \
  $snapshot_name \
  $output_directory/$run_name

python3 performance/particle_updates_step_cost.py \
  $run_name \
  $run_directory \
  $snapshot_name \
  $output_directory/$run_name

python3 performance/wallclock_number_of_steps.py \
  $run_name \
  $run_directory \
  $snapshot_name \
  $output_directory/$run_name

python3 performance/wallclock_simulation_time.py \
  $run_name \
  $run_directory \
  $snapshot_name \
  $output_directory/$run_name