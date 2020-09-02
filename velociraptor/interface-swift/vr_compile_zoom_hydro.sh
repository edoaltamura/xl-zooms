#!/bin/bash -l

# Bash script that configures and compiles VELOCIRAPTOR-STF.

source ../../modules.sh

if [ ! -d ./VELOCIraptor-STF-hydro ]; then
  echo VELOCIraptor-STF source code not found - cloning from GitLab...
  git clone https://github.com/ICRAR/VELOCIraptor-STF
  mv ./VELOCIraptor-STF ./VELOCIraptor-STF-hydro
fi

cd ./VELOCIraptor-STF-hydro
git fetch

# Configure makefile. Compile into executable ./stf
cmake . -DVR_USE_HYDRO=ON -DVR_USE_SWIFT_INTERFACE=ON -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE=Release -DVR_MPI=OFF -DVR_ZOOM_SIM=ON
make -j