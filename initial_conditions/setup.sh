cd modules
export PYTHONPATH=$(pwd)
cythonize -i ./MakeGrid.pyx
cd ..
