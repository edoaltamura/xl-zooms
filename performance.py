from os import system

run_name = ""
run_directory = "/cosma6/data/dp004/rttw52/EAGLE-XL/EAGLE-XL_ClusterSK0_DMO"
snapshot_name = "snapshots/EAGLE-XL_ClusterSK0_DMO_0001.hdf5"
output_path = "outfiles"

performance_scripts = f"""
python3 performance / number_of_steps_simulation_time.py \
    {run_name} \
    {run_directory} \
    {snapshot_name} \
    {output_path}

python3 performance / particle_updates_step_cost.py \
    {run_name} \
    {run_directory} \
    {snapshot_name} \
    {output_path}

python3 performance / wallclock_number_of_steps.py \
    {run_name} \
    {run_directory} \
    {snapshot_name} \
    {output_path}

python3 performance / wallclock_simulation_time.py \
    {run_name} \
    {run_directory} \
    {snapshot_name} \
    {output_path}
"""
system(performance_scripts)
