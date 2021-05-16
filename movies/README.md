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