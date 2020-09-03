#!/bin/bash

source modules.sh

destination_directory=/cosma7/data/dp004/dc-alta2/xl-zooms/hydro
run_name=EAGLE-XL_ClusterSK0

old_directory=$(pwd)

mkdir -p /cosma7/data/dp004/dc-alta2/xl-zooms
mkdir -p $destination_directory
mkdir -p $destination_directory/$run_name

# WE are now in the run data directory
cd $destination_directory/$run_name || exit

# Prepare Velociraptor
if [ ! -d ./VELOCIraptor-STF-hydro ]; then
  echo VELOCIraptor-STF source code not found - cloning from GitLab...
  git clone https://github.com/ICRAR/VELOCIraptor-STF
  mv ./VELOCIraptor-STF ./VELOCIraptor-STF-hydro
fi

# Configure makefile. Compile into executable ./stf
cd ./VELOCIraptor-STF-hydro || exit
git fetch
cmake . -DVR_USE_HYDRO=ON \
  -DVR_USE_SWIFT_INTERFACE=ON \
  -DCMAKE_CXX_FLAGS="-fPIC" \
  -DCMAKE_BUILD_TYPE=Release \
  -DVR_ZOOM_SIM=ON \
  -DVR_MPI=OFF
make -j
cd ..

# Prepare SWIFT
if [ ! -d ./swiftsim-hydro ]; then
  echo SWIFT source code not found - cloning from GitLab...
  git clone https://gitlab.cosma.dur.ac.uk/swift/swiftsim.git
  mv ./swiftsim ./swiftsim-hydro
fi

# Configure makefile
cd ./swiftsim-hydro || exit
sh ./autogen.sh
sh ./configure \
  --with-subgrid=EAGLE-XL \
  --with-hydro=sphenix \
  --with-kernel=wendland-C2 \
  --with-tbbmalloc \
  --enable-ipo
  --with-velociraptor=`pwd`/../VELOCIraptor-STF-hydro/src
make -j
cd ..

mkdir -p ./logs

cp $old_directory/swift/run_hydro.slurm .
cp $old_directory/swift/parameters_hydro.yml .
cp $old_directory/velociraptor/interface-swift/vrconfig_3dfof_subhalos_SO_hydro.cfg .
cp $old_directory/swift/snap_redshifts.txt .
cp $old_directory/swift/vr_redshifts.txt .
cp -r $old_directory/swift/coolingtables .
cp -r $old_directory/swift/yieldtables .
cp /cosma7/data/dp004/rttw52/swift_runs/make_ics/ics/EAGLE-XL_ClusterSK0.hdf5 .

#sbatch ./run_hydro.slurm
squeue -u dc-alta2
cd "$old_directory" || exit
