input_directory="/cosma7/data/dp004/dc-alta2/xl-zooms"

# Choose which cluster and snapshot to run on
simtype="hydro"
haloid=0
resolution="-8"
snap_num=36

run_name=EAGLE-XL_ClusterSK$haloid_$resolutionres
run_directory=$input_directory/$simtype/$run_name/

echo "Removing /stf directory"
rm -rf $run_directory/stf

echo "Replacing the config file"
rm "$run_directory/config/vr_config_zoom_$simtype.cfg"
cp ~/xl-zooms/velociraptor/standalone/vr_config_zoom_hydro.cfg $run_directory/config

echo "Replacing the run file"
rm "$run_directory/config/vr_config_zoom_$simtype.cfg"
cp ~/xl-zooms/velociraptor/standalone/vr_config_zoom_hydro.cfg $run_directory/config

snapshot_name=snapshots/$run_name_00$snap_num.hdf5