#!/bin/bash -l

# Bash script that configures and compiles SWIFT.

source ./modules.sh

old_directory=$(pwd)

if [ ! -d ~/swiftsim ]; then
  echo SWIFT source code not found - cloning from GitLab...
  cd ~
  git clone https://gitlab.cosma.dur.ac.uk/swift/swiftsim.git
  cd $old_directory
fi

cd ~/swiftsim
sh ./autogen.sh
sh ./configure \
  --with-hydro-dimension=2 \
  --with-hydro=sphenix \
  --with-kernel=quintic-spline \
  --disable-hand-vec

# Configure makefile. Compile into executable ./swift
cmake . -DVR_ZOOM_SIM=ON -DCMAKE_BUILD_TYPE=Release -DVR_USE_HYDRO=OFF -DVR_USE_SWIFT_INTERFACE=OFF
make -j

# Copy executable into xl-zooms directory
cd $old_directory
rm ./swift
cp ~/swiftsim/examples/swift .