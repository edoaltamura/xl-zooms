### Movie engine

This sub-directory contains scripts that are set-up for producing movie frames as individual images.
The recommended method for generating these frames is to run individual `python3` instances on single thread and parallelise jobs using the _GNU parallel_ library.

An example launch command:
```shell script
module load gnu-parallel/20181122  # On COSMA systems

parallel python3 my_script.py --arguments argvalues --sequence-number ::: {1..100}
```
or using the `seq` command
```shell script
export x=100
parallel python3 my_script.py --arguments argvalues --sequence-number ::: $(seq 1 $x)
```
---
### Assembling frames
To combine frames on COSMA systems, import and run the `ffmpeg` library as 
```shell script
module load ffmpeg/4.0.2

# If the frames are numbered in sequence
ffmpeg \
    -framerate 60 \
    -start_number 0 \
    -i frame_%04d.png \
    -vf scale=-2:1080,format=yuv420p \
    output_movie.mp4

# If missing frames or not always numbered in sequence
ffmpeg \
    -framerate 20 \
    -pattern_type glob \
    -i 'frame_*.png' \
    -vf scale=-2:1080,format=yuv420p \
    output_movie.mp4
``` 
---
### SLURM launch script
```shell script
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
module load gnu-parallel/20181122
module load openmpi
module load python/3.6.5
module load ffmpeg/4.0.2

parallel python3 my_script.py --arguments argvalues --sequence-number ::: {1..100}

ffmpeg \
    -framerate 3 \
    -pattern_type glob \
    -i 'cooling_times_L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT9_Nheat1_SNnobirth_*.png' \
    -vf scale=-2:1080,format=yuv420p \
    cooling_times_L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT9_Nheat1_SNnobirth.mp4
```