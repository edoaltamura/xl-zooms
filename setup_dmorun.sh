#!/bin/bash

# USAGE:
# Run with: git pull; source setup_dmorun.sh
resolution="-8res"
run_name="EAGLE-XL_ClusterSK2_$resolution"

source modules.sh
destination_directory=/cosma7/data/dp004/dc-alta2/xl-zooms/dmo

old_directory=$(pwd)
mkdir -p /cosma7/data/dp004/dc-alta2/xl-zooms
mkdir -p $destination_directory
mkdir -p $destination_directory/$run_name

# Prepare Velociraptor standlone
cd $destination_directory
if [ ! -d ./VELOCIraptor-STF ]; then
  echo VELOCIraptor-STF source code not found - cloning from GitLab...
  git clone https://github.com/ICRAR/VELOCIraptor-STF
  cd $destination_directory/VELOCIraptor-STF
  git fetch
  cmake . -DVR_USE_HYDRO=OFF \
    -DVR_USE_SWIFT_INTERFACE=OFF \
    -DCMAKE_CXX_FLAGS="-fPIC" \
    -DCMAKE_BUILD_TYPE=Release \
    -DVR_ZOOM_SIM=ON \
    -DVR_MPI=OFF
  make -j
fi

# Prepare SWIFT
cd $destination_directory
if [ ! -d $destination_directory/swiftsim ]; then
  echo SWIFT source code not found - cloning from GitLab...
  git clone https://gitlab.cosma.dur.ac.uk/swift/swiftsim.git
  cd $destination_directory/swiftsim
  sh ./autogen.sh
  sh ./configure \
    --with-tbbmalloc \
    --enable-ipo
  make -j
fi

# We are now in the run data directory
cd $destination_directory/$run_name
mkdir -p ./logs
mkdir -p ./ics
mkdir -p ./config

cp $old_directory/swift/README.md .
cp $old_directory/swift/run_swift.slurm .
cp $old_directory/swift/dmo_$resolution.yml ./config
cp $old_directory/swift/snap_redshifts.txt ./config
cp $old_directory/swift/vr_redshifts.txt ./config
cp $old_directory/velociraptor/standalone/run_vr.slurm .
cp $old_directory/velociraptor/interface-swift/vrconfig_3dfof_subhalos_SO_hydro.cfg ./config
cp /cosma7/data/dp004/rttw52/swift_runs/make_ics/ics/EAGLE-XL_ClusterSK2.hdf5 ./ics
mv ./ics/EAGLE-XL_ClusterSK2.hdf5 ./ics/$run_name.hdf5

# Edit run names in the submission and parameter files
sed -i "s/RUN_NAME/$run_name/" ./run_swift.slurm
sed -i "s/RUN_NAME/$run_name/" ./run_vr.slurm
sed -i "s/RUN_NAME/$run_name/" ./config/dmo_$resolution.yml
sed -i "s/RUN_NAME/$run_name/" ./README.md

sed -i "s/PARAM_FILE/dmo_$resolution/" ./run_swift.slurm
sed -i "s/PARAM_FILE/dmo_$resolution/" ./README.md

sbatch ./run_swift.slurm
cd "$old_directory"
squeue -u dc-alta2
