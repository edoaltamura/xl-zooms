#!/bin/bash

# USAGE
# Run with: git pull; source migrate_source.sh

run_name=EAGLE-XL_ClusterSK2

source modules.sh
destination_directory=/cosma7/data/dp004/dc-alta2/xl-zooms/hydro

old_directory=$(pwd)

mkdir -p /cosma7/data/dp004/dc-alta2/xl-zooms
mkdir -p $destination_directory
mkdir -p $destination_directory/$run_name

# WE are now in the run data directory
cd $destination_directory/$run_name || exit

# Prepare Velociraptor standlone
#if [ ! -d ./VELOCIraptor-STF-hydro ]; then
#  echo VELOCIraptor-STF source code not found - cloning from GitLab...
#  git clone https://github.com/ICRAR/VELOCIraptor-STF
#  mv ./VELOCIraptor-STF ./VELOCIraptor-STF-hydro
#fi
#
#cd ./VELOCIraptor-STF-hydro || exit
#git fetch
#cmake . -DVR_USE_HYDRO=ON \
#  -DVR_USE_SWIFT_INTERFACE=OFF \
#  -DCMAKE_CXX_FLAGS="-fPIC" \
#  -DCMAKE_BUILD_TYPE=Release \
#  -DVR_ZOOM_SIM=ON \
#  -DVR_MPI=OFF
#make -j
#cd ..
#
## Prepare SWIFT
#if [ ! -d ./swiftsim-hydro ]; then
#  echo SWIFT source code not found - cloning from GitLab...
#  git clone https://gitlab.cosma.dur.ac.uk/swift/swiftsim.git
#  mv ./swiftsim ./swiftsim-hydro
#fi
#
#cd ./swiftsim-hydro || exit
#sh ./autogen.sh
#sh ./configure \
#  --with-subgrid=EAGLE-XL \
#  --with-hydro=sphenix \
#  --with-kernel=wendland-C2 \
#  --with-tbbmalloc \
#  --enable-ipo
#make -j
#cd ..

mkdir -p ./logs
mkdir -p ./config
mkdir -p ./ics

cp $old_directory/swift/run_swift.slurm .
cp $old_directory/modules.sh .
cp $old_directory/velociraptor/standalone/run_vr.slurm .
cp $old_directory/swift/eagle_-8res.yml ./config
cp $old_directory/velociraptor/interface-swift/vrconfig_3dfof_subhalos_SO_hydro.cfg ./config
cp $old_directory/swift/snap_redshifts.txt ./config
cp $old_directory/swift/vr_redshifts.txt ./config
cp /cosma7/data/dp004/rttw52/swift_runs/make_ics/ics/$run_name.hdf5 ./ics
cp -r $old_directory/swift/coolingtables .
cp -r $old_directory/swift/yieldtables .
cp /cosma7/data/dp004/dc-ploe1/CoolingTables/2019_most_recent/UV_dust1_CR1_G1_shield1.hdf5 .
cp $old_directory/swift/README.md .

# Edit run names in the submission and parameter files
sed -i "s/RUN_NAME/$run_name/" ./run_swift.slurm
sed -i "s/RUN_NAME/$run_name/" ./run_vr.slurm
sed -i "s/RUN_NAME/$run_name/" ./config/eagle_-8res.yml
sed -i "s/RUN_NAME/$run_name/" ./README.md

sbatch ./run_swift.slurm
squeue -u dc-alta2
cd "$old_directory" || exit