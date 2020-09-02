#!/bin/bash -l

# Bash script that configures and compiles SWIFT.

source ../modules.sh

if [ ! -d ./swiftsim-hydro ]; then
  echo SWIFT source code not found - cloning from GitLab...
  git clone https://gitlab.cosma.dur.ac.uk/swift/swiftsim.git
  mv ./swiftsim ./swiftsim-hydro
fi

# Configure makefile
cd ./swiftsim-hydro
sh ./autogen.sh
sh ./configure \
  --with-subgrid=EAGLE-XL \
  --with-hydro=sphenix \
  --with-kernel=wendland-C2 \
  --with-tbbmalloc \
  --enable-ipo
#  --with-velociraptor=`pwd`/../velociraptor/interface-swift/VELOCIraptor-STF-hydro/src

make -j
cd ..