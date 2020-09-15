cd modules
export PYTHONPATH=$(pwd)
cd ..
cythonize -i ./MakeGrid.pyx