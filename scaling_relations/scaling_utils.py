# Plot scaling relations for EAGLE-XL tests
import sys
import os
import pandas as pd
from typing import Callable, List
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

# Make the register backend visible to the script
sys.path.append("../zooms")
sys.path.append("../observational_data")

from register import zooms_register, Zoom, Tcut_halogas, name_list

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass


def set_output_names(dataset_names: List[str]):
    """
    Decorator used to specify names for the dataset columns in the
    Pandas dataframe. The dataset names should be representative
    of the variable names used in the code.
    See e.g. `columns = _process_single_halo.dataset_names` assignment
    in `process_catalogue`.
    """
    def wrapper(func):
        setattr(func, 'dataset_names', dataset_names)
        return func

    return wrapper


def set_scaling_relation_name(scaling_relation_name: str):
    """
    Decorator used to specify the name of the scaling relation in
    use. This name is called e.g. in the case you want to save
    intermediate outputs and will become the filename of the
    output file. Usually, it is set as the name of the script
    file used for generating the scaling relation data.
    See `if save_dataframe:` block in `process_catalogue`.
    """
    def wrapper(func):
        setattr(func, 'scaling_relation_name', scaling_relation_name)
        return func

    return wrapper


def process_catalogue(_process_single_halo: Callable, find_keyword: str = '',
                      save_dataframe: bool = False) -> pd.DataFrame:
    """
    This function performs the collective multi-threaded I/O for processing
    the halos in the catalogue. It can accept different types of function
    through the `_process_single_halo` argument and can even be used to
    compute profiles in parallel. Note that this implementation allocates
    one catalogue member to each thread and reflects embarrassingly
    parallel jobs. It cannot be used in distributed HPC with MPI and
    can only be run on a single node. Make sure the memory isn't filled -
    in such case you will need a distributed MPI version.

    @args find_keyword: specifies a keyword to filter the zooms register by.

    @args save_dataframe: set to True if you want to save the intermediate
    output of the calculation. Currently using pickle files generated by
    Pandas.
    """
    if not find_keyword:
        # If find_keyword is empty, collect all zooms
        _zooms_register = zooms_register
    else:
        _zooms_register = [zoom for zoom in zooms_register if find_keyword in zoom.run_name]

    _name_list = [zoom.run_name for zoom in _zooms_register]

    if len(_zooms_register) == 1:
        print("Analysing one object only. Not using multiprocessing features.")
        results = [_process_single_halo(_zooms_register[0])]
    else:
        num_threads = len(_zooms_register) if len(_zooms_register) < cpu_count() else cpu_count()
        print(f"Analysis of {len(_zooms_register):d} zooms mapped onto {num_threads:d} CPUs.")

        # The results of the multiprocessing Pool are returned in the same order as inputs
        with Pool(num_threads) as pool:
            results = pool.map(_process_single_halo, iter(_zooms_register))

    # Recast output into a Pandas dataframe for further manipulation
    columns = _process_single_halo.dataset_names
    results = pd.DataFrame(list(results), columns=columns)
    results.insert(0, 'Run name', pd.Series(_name_list, dtype=str))
    print(results.head())

    if save_dataframe:
        file_name = os.path.join(
            f'{zooms_register[0].output_directory}',
            f'{_process_single_halo.scaling_relation_name}.pkl'
        )
        results.to_pickle(file_name)
        print(f"Catalogue file saved to {file_name}")

    return results
