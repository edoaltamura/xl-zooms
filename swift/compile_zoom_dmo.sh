#!/bin/bash -l

# Bash script that configures and compiles SWIFT.

source ../modules.sh

echo Preparing VELOCIraptor code...
sh ../velociraptor/interface-swift/compile_zoom_dmo.sh

if [ ! -d ./swiftsim-dmo ]; then
  echo SWIFT source code not found - cloning from GitLab...
  git clone https://gitlab.cosma.dur.ac.uk/swift/swiftsim.git
  mv ./swiftsim ./swiftsim-dmo
fi

# Configure makefile
cd ./swiftsim-dmo
sh ./autogen.sh
sh ./configure \
  --with-tbbmalloc \
  --enable-ipo \
  --with-velociraptor=../velociraptor/interface-swift/VELOCIraptor-STF-dmo/src

make -j
cd ..