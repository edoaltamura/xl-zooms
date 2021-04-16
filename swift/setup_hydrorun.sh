#!/bin/bash

# USAGE:
# Run with: git pull; source setup_hydrorun.sh

resolution="-8res"
run_name="EAGLE-XL_ClusterSK0_$resolution"

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
  cd $destination_directory/VELOCIraptor-STF
  git fetch
  cmake . -DVR_USE_HYDRO=ON \
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
    --with-subgrid=EAGLE-XL \
    --with-hydro=sphenix \
    --with-kernel=wendland-C2 \
    --with-tbbmalloc \
    --enable-ipo
  make -j
fi

# Retrieve cool/yield tables
cd $destination_directory
if [ ! -d ./yieldtables ]; then
  wget http://virgodb.cosma.dur.ac.uk/swift-webstorage/YieldTables/EAGLE/yieldtables.tar.gz
  tar -xf yieldtables.tar.gz
  rm ./yieldtables.tar.gz
fi
if [ ! -d ./coolingtables ]; then
  wget http://virgodb.cosma.dur.ac.uk/swift-webstorage/CoolingTables/EAGLE/coolingtables.tar.gz
  tar -xf coolingtables.tar.gz
  rm ./coolingtables.tar.gz
  wget http://virgodb.cosma.dur.ac.uk/swift-webstorage/CoolingTables/COLIBRE/UV_dust1_CR1_G1_shield1.hdf5
fi

# We are now in the run data directory
cd $destination_directory/$run_name
mkdir -p ./logs
mkdir -p ./ics
mkdir -p ./config

cp $old_directory/swift/README.md .
cp $old_directory/swift/run_swift_hydro.slurm .
cp $old_directory/swift/eagle_$resolution.yml ./config
cp $old_directory/swift/snap_redshifts.txt ./config
cp $old_directory/swift/vr_redshifts.txt ./config
cp $old_directory/velociraptor/standalone/run_vr.slurm .
cp $old_directory/velociraptor/standalone/vr_config_zoom_hydro.cfg ./config
cp /cosma7/data/dp004/rttw52/swift_runs/make_ics/ics/EAGLE-XL_ClusterSK0_High.hdf5 ./ics
mv ./ics/EAGLE-XL_ClusterSK0_High.hdf5 ./ics/$run_name.hdf5
mv ./run_swift_hydro.slurm ./run_swift.slurm

# Edit run names in the submission and parameter files
sed -i "s/RUN_NAME/$run_name/" ./run_swift.slurm
sed -i "s/RUN_NAME/$run_name/" ./run_vr.slurm
sed -i "s/RUN_NAME/$run_name/" ./config/eagle_+1res.yml
sed -i "s/RUN_NAME/$run_name/" ./README.md

sed -i "s/PARAM_FILE/eagle_$resolution/" ./run_swift.slurm
sed -i "s/PARAM_FILE/eagle_$resolution/" ./README.md

sbatch ./run_swift.slurm
cd "$old_directory"
squeue -u dc-alta2
