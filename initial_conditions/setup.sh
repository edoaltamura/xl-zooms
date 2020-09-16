cd modules
export PYTHONPATH=$(pwd)
cd ../particle_load
cythonize -i ./MakeGrid.pyx
cd ..
