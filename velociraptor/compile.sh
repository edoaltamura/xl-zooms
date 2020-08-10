#!/bin/bash -l

# Bash script that configures and compiles VELOCIRAPTOR-STF.

source ./modules.sh

old_directory=$(pwd)

if [ ! -d ~/VELOCIraptor-STF ]; then
  echo VELOCIraptor-STF source code not found - cloning from GitLab...
  cd ~
  git clone https://github.com/pelahi/VELOCIraptor-STF
  cd $old_directory
fi

cd ~/VELOCIraptor-STF
git fetch

# Configure makefile. Compile into executable ./stf
cmake . -DVR_ZOOM_SIM=ON -DCMAKE_BUILD_TYPE=Release
make -j

# Copy executable into xl-zooms directory
cd $old_directory
rm ./stf
cp ~/VELOCIraptor-STF/stf .