#!/bin/bash

# USAGE:
# Run with: git pull; source migrate_source.sh

run_name=EAGLE-XL_ClusterSK0

source modules.sh
destination_directory=/cosma7/data/dp004/dc-alta2/xl-zooms/hydro

old_directory=$(pwd)
mkdir -p /cosma7/data/dp004/dc-alta2/xl-zooms
mkdir -p $destination_directory
mkdir -p $destination_directory/$run_name

# Prepare Velociraptor standlone
cd $destination_directory
if [ ! -d ./VELOCIraptor-STF ]; then
  echo VELOCIraptor-STF source code not found - cloning from GitLab...
  git clone https://github.com/ICRAR/VELOCIraptor-STF
fi

cd $destination_directory/VELOCIraptor-STF
git fetch
cmake . -DVR_USE_HYDRO=ON \
  -DVR_USE_SWIFT_INTERFACE=OFF \
  -DCMAKE_CXX_FLAGS="-fPIC" \
  -DCMAKE_BUILD_TYPE=Release \
  -DVR_ZOOM_SIM=ON \
  -DVR_MPI=OFF
make -j
vr_path_exe=$destination_directory/VELOCIraptor-STF

# Prepare SWIFT
cd $destination_directory
if [ ! -d $destination_directory/swiftsim ]; then
  echo SWIFT source code not found - cloning from GitLab...
  git clone https://gitlab.cosma.dur.ac.uk/swift/swiftsim.git
fi

cd $destination_directory/swiftsim
sh ./autogen.sh
sh ./configure \
  --with-subgrid=EAGLE-XL \
  --with-hydro=sphenix \
  --with-kernel=wendland-C2 \
  --with-tbbmalloc \
  --enable-ipo \
  --with-parmetis \
  --with-gsl \
  --enable-debug
make -j
swift_path_exe=$destination_directory/swiftsim/examples


# We are now in the run data directory
cd $destination_directory/$run_name
mkdir -p ./logs
mkdir -p ./ics

cp $old_directory/swift/README.md .
cp $old_directory/swift/run_swift.slurm .
cp $old_directory/swift/eagle_-8res.yml .
cp $old_directory/swift/snap_redshifts.txt .
cp $old_directory/swift/vr_redshifts.txt .
cp -r $old_directory/swift/coolingtables .
cp -r $old_directory/swift/yieldtables .
cp $old_directory/velociraptor/standalone/run_vr.slurm .
cp $old_directory/velociraptor/interface-swift/vrconfig_3dfof_subhalos_SO_hydro.cfg .
cp /cosma7/data/dp004/rttw52/swift_runs/make_ics/ics/$run_name.hdf5 ./ics
cp /cosma7/data/dp004/dc-ploe1/CoolingTables/2019_most_recent/UV_dust1_CR1_G1_shield1.hdf5 .

# Edit run names in the submission and parameter files
sed -i "s/RUN_NAME/$run_name/" ./run_swift.slurm
sed -i "s/RUN_NAME/$run_name/" ./run_vr.slurm
sed -i "s/RUN_NAME/$run_name/" ./eagle_-8res.yml
sed -i "s/RUN_NAME/$run_name/" ./README.md
sed -i "s/PATH_EXECUTABLE/$swift_path_exe/" ./run_swift.slurm
sed -i "s/PATH_EXECUTABLE/$vr_path_exe/" ./run_vr.slurm

sbatch ./run_swift.slurm
cd "$old_directory"
squeue -u dc-alta2
