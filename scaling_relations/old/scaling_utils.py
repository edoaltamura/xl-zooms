# Plot scaling relations for EAGLE-XL tests
import sys
import os
import pandas as pd
from typing import Callable, List, Union
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor

# Make the register backend visible to the script
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            'observational_data'
        )
    )
)
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            'zooms'
        )
    )
)

from register import zooms_register, Zoom, calibration_zooms
from auto_parser import args, parser


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


def check_catalogue_completeness(
        results_catalogue_path: str,
        find_keyword: Union[list, str] = None,
) -> bool:
    """

    """
    if find_keyword is None:
        # If find_keyword is empty, collect all zooms
        _zooms_register = zooms_register

    elif type(find_keyword) is str:
        _zooms_register = [zoom for zoom in zooms_register if find_keyword in zoom.run_name]

    elif type(find_keyword) is list:
        _zooms_register = []
        for keyword in find_keyword:
            for zoom in zooms_register:
                if keyword in zoom.run_name and zoom not in _zooms_register:
                    _zooms_register.append(zoom)

    _name_list = [zoom.run_name for zoom in _zooms_register]

    # Import catalogue from pickle file
    results_catalogue = pd.read_pickle(results_catalogue_path)
    catalogue_name_list = results_catalogue['Run name'].values.tolist()

    # Check if all results in query are already in the computed catalogue
    completeness = all(element in catalogue_name_list for element in _name_list)

    if completeness:
        print(f"All objects in original query are already in the computed catalogue. {len(catalogue_name_list)} found.")
    else:
        missing_zooms = [name for name in _name_list if name not in catalogue_name_list]
        print((
            f"Not all objects in original query are already in the computed catalogue. "
            f"Query has {len(_name_list)} zooms. "
            f"Computed catalogue has {len(catalogue_name_list)} zooms. "
            f"Missing zooms:\n{missing_zooms}"
        ))

    return completeness


def process_catalogue(_process_single_halo: Callable,
                      find_keyword: Union[list, str] = None,
                      save_dataframe: bool = False,
                      concurrent_threading: bool = False,
                      no_multithreading: bool = False) -> pd.DataFrame:
    """
    This function performs the collective multi-threaded I/O for processing
    the halos in the catalogue. It can accept different types of function
    through the `_process_single_halo` argument and can even be used to
    compute profiles in parallel. Note that this implementation allocates
    one catalogue member to each thread and reflects embarrassingly
    parallel jobs. It cannot be used in distributed HPC with MPI and
    can only be run on a single node. Make sure the memory isn't filled -
    in such case you will need a distributed MPI version.

    @xlargs find_keyword: specifies a keyword to filter the zooms register by.

    @xlargs save_dataframe: set to True if you want to save the intermediate
    output of the calculation. Currently using pickle files generated by
    Pandas.
    """
    # Print the CLI arguments that are parsed in the script
    for parsed_argument in vars(args):
        print(f"{parsed_argument} = {getattr(args, parsed_argument)}")

    print(f"z = {calibration_zooms.redshift_from_index(args.redshift_index):.2f}")

    if find_keyword is None:
        # If find_keyword is empty, collect all zooms
        _zooms_register = zooms_register

    elif type(find_keyword) is str:
        _zooms_register = [zoom for zoom in zooms_register if find_keyword in zoom.run_name]

    elif type(find_keyword) is list:
        _zooms_register = []
        for keyword in find_keyword:
            for zoom in zooms_register:
                if keyword in zoom.run_name and zoom not in _zooms_register:
                    _zooms_register.append(zoom)

    _name_list = [zoom.run_name for zoom in _zooms_register]

    if len(_zooms_register) == 1:
        print("Analysing one object only. Not using multiprocessing features.")
        results = [_process_single_halo(_zooms_register[0])]

    else:

        if no_multithreading:
            print(f"Running with no multithreading.\nAnalysing {len(_zooms_register):d} zooms serially.")
            results = []
            for i, zoom in enumerate(_zooms_register):
                print(f"({i + 1}/{len(_zooms_register)}) Processing: {zoom.run_name}")
                results.append(
                    _process_single_halo(zoom)
                )

        else:

            print("Running with multithreading.")
            num_threads = len(_zooms_register) if len(_zooms_register) < cpu_count() else cpu_count()
            print(f"Analysis of {len(_zooms_register):d} zooms mapped onto {num_threads:d} CPUs.")

            threading_engine = Pool(num_threads)
            if concurrent_threading:
                threading_engine = ProcessPoolExecutor(max_workers=num_threads)

            try:
                # The results of the multiprocessing Pool are returned in the same order as inputs
                with threading_engine as pool:
                    results = pool.map(_process_single_halo, iter(_zooms_register))
            except Exception as error:
                print((
                    f"The analysis stopped due to the error\n{error}\n"
                    "Please use a different multiprocessing pool or run the code serially."
                ))
                raise error

    # Recast output into a Pandas dataframe for further manipulation
    columns = _process_single_halo.dataset_names
    results = pd.DataFrame(list(results), columns=columns)
    results.insert(0, 'Run name', pd.Series(_name_list, dtype=str))
    if not args.quiet:
        print(results.head())

    if save_dataframe:
        file_name = os.path.join(
            f'{zooms_register[0].output_directory}',
            f'{_process_single_halo.scaling_relation_name}'
        )

        # Save data in pickle format (saves python object)
        results.to_pickle(f'{file_name}.pkl')

        # Save data in text format (useful for consulting)
        results.to_csv(f'{file_name}.txt', header=True, index=False, sep='\t', mode='w')

        print(f"Catalogue file saved to {file_name}(.pkl/.txt)")

    return results
