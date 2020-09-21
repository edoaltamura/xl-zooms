cd modules
export PYTHONPATH=$(pwd)
ls -lh
cd ../particle_load
cythonize -i ./MakeGrid.pyx
cd ..
