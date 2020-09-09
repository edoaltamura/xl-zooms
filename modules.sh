#!/bin/bash -l

echo Loading modules for swiftsim...
module purge
module load cosma/2018
module load intel_comp/2018
module load intel_mpi/2018
module load parmetis/4.0.3
module load parallel_hdf5/1.8.20
module load gsl/2.4
module load fftw/3.3.7
module load cmake
module load python/3.6.5
module load ffmpeg/4.0.2
module load llvm/7.0.0
module load allinea/ddt/20.0.3
module load utils/201805
module load hdfview/2.14.0
module load gadgetviewer/1.1.0
echo All modules loaded.