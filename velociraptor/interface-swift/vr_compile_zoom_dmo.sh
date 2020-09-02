#!/bin/bash -l

# Bash script that configures and compiles VELOCIRAPTOR-STF.

source ../../modules.sh

if [ ! -d ./VELOCIraptor-STF-dmo ]; then
  echo VELOCIraptor-STF source code not found - cloning from GitLab...
  git clone https://github.com/pelahi/VELOCIraptor-STF
  mv ./VELOCIraptor-STF ./VELOCIraptor-STF-dmo
fi

cd ./VELOCIraptor-STF-dmo
git fetch

# Configure makefile. Compile into executable ./stf
cmake . -DVR_USE_HYDRO=OFF -DVR_USE_SWIFT_INTERFACE=ON -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE=Release -DVR_MPI=OFF -DVR_ZOOM_SIM=ON
make -j