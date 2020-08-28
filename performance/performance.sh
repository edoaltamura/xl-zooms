run_name="Zooms"
run_directory="/cosma6/data/dp004/rttw52/EAGLE-XL/EAGLE-XL_ClusterSK1_DMO"
snapshot_name="snapshots/EAGLE-XL_ClusterSK1_DMO_0001.hdf5"
output_path="outfiles"

python3 number_of_steps_simulation_time.py \
  $run_name \
  $run_directory \
  $snapshot_name \
  $output_path

python3 particle_updates_step_cost.py \
  $run_name \
  $run_directory \
  $snapshot_name \
  $output_path

python3 wallclock_number_of_steps.py \
  $run_name \
  $run_directory \
  $snapshot_name \
  $output_path

python3 wallclock_simulation_time.py \
  $run_name \
  $run_directory \
  $snapshot_name \
  $output_path