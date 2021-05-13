#!/bin/bash -l

#SBATCH --ntasks 28
#SBATCH -J cooling_times
#SBATCH -o standard_output_file.%J.out
#SBATCH -e standard_error_file.%J.err
#SBATCH -p cosma7-prince
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 10:00:00

module purge
module load gnu_comp
module load openmpi
module load python/3.6.5
module load ffmpeg/4.0.2

mpirun -np 12 python3 cooling_times_movie.py -a 15 -q

#ffmpeg \
#    -framerate 60 \
#    -start_number 0 \
#    -i /cosma/home/dp004/dc-alta2/data7/xl-zooms/analysis/cooling_times_L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8p5_Nheat1_SNnobirth_%04d.png \
#    -vf scale=-2:1080,format=yuv420p \
#    /cosma/home/dp004/dc-alta2/data7/xl-zooms/analysis/cooling_times_L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8p5_Nheat1_SNnobirth.mp4


ffmpeg \
    -framerate 10 \
    -pattern_type glob \
    -i 'cooling_times_L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8p5_Nheat1_SNnobirth_*.png' \
    -vf scale=-2:1080,format=yuv420p \
    cooling_times_L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8p5_Nheat1_SNnobirth.mp4