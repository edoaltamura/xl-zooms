# -*- coding: utf-8 -*-
"""Register back-end for EAGLE-XL Zoom calibration

This module contains the classes and handles for retrieving the zoom simulations
throughout the COSMA system, by specifying some _look-up_ directories.
The main use of this module includes the interoperability with higher-level
modules and API for the analysis pipeline.

Example:
    Running the file will print the current status of the simulations, which can
    used as a summary list for all runs generated and for their locations in the
    system. Remember to `git pull` changes from the remote repository regularly.

        $ python3 register.py

    To use the functionalities in `register.py`, use the import statement or
    import specific attributes/classes.

        import register
        from register import zooms_register

    From this you can access zoom-specific data, such as

         zooms_register[0].run_name

Attributes:
    Tcut_halogas (float): Hot gas temperature threshold in Kelvin. The analysis
    will only consider gas particles above this temperature, excluding cold gas
    and gas on the equation of state.

    calibration_zooms (EXLZooms): class instance which can be used to retrieve
    info directly from the base class or to call some useful static methods.

    zooms_register (List[Zoom]): specifies the zooms catalogues formed with
    runs that have completed. These are sorted by VR number.

    name_list (List[str]): collects the names of the runs that have completed
    and is specular to `zooms_register[:].run_name`. These are sorted by VR
    number.

Todo:
    * None


To maintain a readable and extensible documentation, please refer to this
guidelines:
.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import io
import h5py
import psutil
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
import subprocess as sp

from .auto_parser import xlargs
from .static_parameters import *
from .intermediate_io import MultiObjPickler


def dump_memory_usage() -> None:
    total_memory = psutil.virtual_memory().total
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss  # in bytes
    print((
        f"[Resources] Memory usage: {memory / 1024 / 1024:.2f} MB "
        f"({memory / total_memory * 100:.2f}%)"
    ))


def tail(fname, window=2):
    """Read last N lines from file fname."""
    if window <= 0:
        raise ValueError('invalid window value %r' % window)
    with open(fname, 'rb') as f:
        BUFSIZ = 1024
        # True if open() was overridden and file was opened in text
        # mode. In that case readlines() will return unicode strings
        # instead of bytes.
        encoded = getattr(f, 'encoding', False)
        CR = '\n' if encoded else b'\n'
        data = '' if encoded else b''
        f.seek(0, os.SEEK_END)
        fsize = f.tell()
        block = -1
        exit = False
        while not exit:
            step = (block * BUFSIZ)
            if abs(step) >= fsize:
                f.seek(0)
                newdata = f.read(BUFSIZ - (abs(step) - fsize))
                exit = True
            else:
                f.seek(step, os.SEEK_END)
                newdata = f.read(BUFSIZ)
            data = newdata + data
            if data.count(CR) >= window:
                break
            else:
                block -= 1
        return data.splitlines()[-window:]


class EXLZooms(object):
    # Zooms will be searched in this directories
    cosma_repositories: List[str] = [
        "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro",
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro",
        "/snap7/scratch/dp004/dc-alta2/xl-zooms/hydro",
    ]

    def __init__(self) -> None:

        # Initialise variables as instance attributes, not as class attributes
        self.name: str = 'Eagle-XL Zooms'
        self.output_directory: str = default_output_directory
        self.name_list: List[str] = []
        self.run_directories: List[str] = []
        self.complete_runs: np.ndarray

        # Search for any run directories in repositories
        for repository in self.cosma_repositories:
            for run_basename in os.listdir(repository):
                run_abspath = os.path.join(repository, run_basename)
                if os.path.isdir(run_abspath) and run_basename.startswith('L0300N0564'):
                    self.run_directories.append(run_abspath)
                    self.name_list.append(run_basename)

        # Classify complete and incomplete runs
        self.complete_runs = np.zeros(len(self.name_list), dtype=np.bool)

        for i, run_directory in enumerate(self.run_directories):
            snaps_path = os.path.join(run_directory, 'snapshots')
            catalogues_path = os.path.join(run_directory, 'stf')

            if os.path.isdir(snaps_path):
                number_snapshots = len([file for file in os.listdir(snaps_path) if file.endswith('.hdf5')])
            else:
                number_snapshots = 0

            if os.path.isdir(catalogues_path):
                number_catalogues = len([subdir for subdir in os.listdir(catalogues_path)])
            else:
                number_catalogues = 0

            # Check that there are as many outputs as in the output_list
            output_list_file = os.path.join(run_directory, 'snap_redshifts.txt')
            output_list = pd.read_csv(output_list_file)

            if " Select Output" in output_list.columns:
                number_snapshots_in_outputlist = np.logical_or.reduce(
                        [output_list[" Select Output"] == f" Snapshot"]
                    ).sum()
            else:
                number_snapshots_in_outputlist = len(output_list["# Redshift"].values)

            if (
                    (number_snapshots > 0) and
                    (number_catalogues > 0) and
                    (number_snapshots == number_catalogues) and
                    (number_snapshots == number_snapshots_in_outputlist)
            ):
                self.complete_runs[i] = True

        if __name__ == "__main__":
            print(f"Found {len(self.name_list):d} zoom directories.")
            print(f"Runs completed: {self.complete_runs.sum():d}")
            print(f"Runs not completed: {len(self.complete_runs) - self.complete_runs.sum():d}")

    @staticmethod
    def get_vr_number_from_name(basename: str) -> int:
        """
        Takes a zoom run base-name and returns the Velociraptor catalogue number
        contained in it. Accepts strings and returns the index as integer.

        Example:
            Accepts: str L0300N0564_VR187_-8res_Isotropic_fixedAGNdT8_Nheat1_SNnobirth
            Returns: int 187

        Args:
            basename (str): The basename of the run directory (not absolute path).

        Returns:
            int: The Velociraptor catalogue number derived from the basename
        """
        start = 'VR'
        end = '_'
        start_position = basename.find(start) + len(start)
        result = basename[start_position:basename.find(end, start_position)]
        return int(result)

    def get_vr_numbers_unique(self) -> List[int]:
        """
        Accessed the full name_list attribute from the class and returns a list with
        the unique VR numbers in the catalogue.

        Returns:
            List[int]: List of unique VR numbers in the catalogue
        """
        vr_nums = [self.get_vr_number_from_name(name) for name in self.name_list]
        vr_nums = set(vr_nums)
        return list(vr_nums).sort()

    def get_completed_run_directories(self) -> List[str]:
        directories = np.array(self.run_directories, dtype=np.str)
        return directories[self.complete_runs].tolist()

    def get_incomplete_run_directories(self) -> List[str]:
        directories = np.array(self.run_directories, dtype=np.str)
        return directories[~self.complete_runs].tolist()

    def get_completed_run_names(self) -> List[str]:
        directories = np.array(self.name_list, dtype=np.str)
        return directories[self.complete_runs].tolist()

    def get_incomplete_run_names(self) -> List[str]:
        directories = np.array(self.name_list, dtype=np.str)
        return directories[~self.complete_runs].tolist()

    def get_snap_redshifts(self, directory: str = None):
        if directory is None:
            run_directory = self.get_completed_run_directories()[0]
        else:
            run_directory = directory

        output_list_file = os.path.join(run_directory, 'snap_redshifts.txt')
        output_list = pd.read_csv(output_list_file)
        redshifts = output_list["# Redshift"].values

        if " Select Output" in output_list.columns:
            index_snaps = np.arange(len(output_list))[
                np.logical_or.reduce(
                    [output_list[" Select Output"] == f" Snapshot"]
                )
            ]
        else:
            index_snaps = np.arange(len(redshifts))

        return redshifts[index_snaps]

    def redshift_from_index(self, redshift_index: int):
        redshifts = self.get_snap_redshifts()
        return redshifts[redshift_index]

    @staticmethod
    def query_slurm() -> List[str]:
        status_code = ["PD", "R", "CG", "S", "ST"]
        status_description = ["pending", "running", "compl", "susp", "stop"]
        slurm_status_descriptor = dict(zip(status_code, status_description))

        cmd = os.path.expandvars("squeue -u $USER -o '%j %t'")
        piper = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)

        # slurp off header line
        jobs = piper.stdout.readlines()[1:]

        return [job_info.decode("utf-8") for job_info in jobs]

    def analyse_incomplete_runs(self, stdout: bool = True) -> pd.DataFrame:

        incomplete_name_list = self.get_incomplete_run_names()
        incomplete_run_directories = self.get_incomplete_run_directories()
        slurm_jobs = self.query_slurm()

        analysis_criteria = [
            'Run name',
            'Run directory',
            'Time steps log found',
            'Last redshift',
            'Snapshots initialised',
            'Snapshots number',
            'Snapshots number target',
            'Stf initialised',
            'Stf number',
            'SLURM SWIFT running',
            'SLURM SWIFT queuing'
        ]

        incomplete_analysis = pd.DataFrame(columns=analysis_criteria)

        for i, (run_name, run_directory) in enumerate(zip(incomplete_name_list, incomplete_run_directories)):

            # Initialise default analysis values
            timesteps_file_found = False
            last_redshift = np.inf
            number_snapshots = 0
            number_catalogues = 0

            for file in os.listdir(run_directory):

                if file.startswith('timesteps_'):
                    timesteps_file_found = True

                    # Get the last redshift printed to the timesteps logs
                    timesteps_file = os.path.join(run_directory, file)
                    lastlast_line, last_line = tail(timesteps_file, window=2)
                    lastlast_line = lastlast_line.split()
                    last_line = last_line.split()

                    if len(lastlast_line) == len(last_line):
                        last_redshift = float(last_line[3])
                    elif len(lastlast_line) > len(last_line):
                        last_redshift = float(lastlast_line[3])

                    break

            # Analyse the snapshot and stf directories
            snaps_path = os.path.join(run_directory, 'snapshots')
            catalogues_path = os.path.join(run_directory, 'stf')
            snapshot_dir_found = os.path.isdir(snaps_path)
            stf_dir_found = os.path.isdir(catalogues_path)

            if snapshot_dir_found:
                number_snapshots = len([file for file in os.listdir(snaps_path) if file.endswith('.hdf5')])
            else:
                number_snapshots = 0

            if stf_dir_found:
                number_catalogues = len([subdir for subdir in os.listdir(catalogues_path)])
            else:
                number_catalogues = 0

            redshifts = self.get_snap_redshifts(directory=run_directory)
            number_snapshots_target = len(redshifts)

            # Analyse queries from SLURM
            slurm_swift_running = False
            slurm_swift_queuing = False

            for line in slurm_jobs:
                job_name, job_status = line.strip().split()
                if job_name == run_name:
                    if job_status == 'PD':
                        slurm_swift_queuing = True
                    elif job_status == 'R':
                        slurm_swift_running = True

                    break

            incomplete_analysis.loc[i] = [
                run_name,
                run_directory,
                timesteps_file_found,
                last_redshift,
                snapshot_dir_found,
                number_snapshots,
                number_snapshots_target,
                stf_dir_found,
                number_catalogues,
                slurm_swift_running,
                slurm_swift_queuing
            ]

        # Sort dataframe by SLURM status and reached redshift
        incomplete_analysis.sort_values(
            by=['SLURM SWIFT running', 'SLURM SWIFT queuing', 'Last redshift'],
            ascending=False,
            inplace=True
        )

        if stdout:
            print((
                f"\n{' Detailed search on incomplete runs ':-^40s}\n"
                f"{' Legend ':-^40s}\n"
                "`!` = Incomplete \n`SW` = SWIFT \n`VR` = Velociraptor \n"
                "`z = 127.` = Simulation not started \n"
                "`xxx` = Not on SLURM queue \n`que` = Pending in SLURM queue \n"
                "`run` = Running on partition\n"
                f"{'':-^40s}\n"
            ))
            incomplete_analysis_listform = incomplete_analysis.values.tolist()
            for i in range(len(incomplete_analysis_listform)):
                (
                    run_name,
                    run_directory,
                    timesteps_file_found,
                    last_redshift,
                    snapshot_dir_found,
                    number_snapshots,
                    number_snapshots_target,
                    stf_dir_found,
                    number_catalogues,
                    slurm_swift_running,
                    slurm_swift_queuing
                ) = incomplete_analysis_listform[i]

                slurm_code = 'xxx'
                if slurm_swift_running:
                    slurm_code = 'run'
                elif slurm_swift_queuing:
                    slurm_code = 'que'

                if np.isinf(last_redshift):
                    last_redshift_label = '127.'
                else:
                    last_redshift_label = f"{last_redshift:3.2f}"

                print((
                    f"[{'!' if last_redshift > 0. else ' '}SW] "
                    f"[z = {last_redshift_label:s}] "
                    f"[{'!' if number_catalogues < number_snapshots_target else ' '}VR] "
                    f"[SLURM {slurm_code:s}] "
                    f"{run_name}"
                ))

        return incomplete_analysis


class Redshift(object):
    __slots__ = (
        'run_name',
        'scale_factor',
        'a',
        'redshift',
        'z',
        'snapshot_path',
        'catalogue_properties_path',
    )

    def __init__(self, info_dict: dict):
        # Initialise variables as instance attributes, not as class attributes
        self.run_name: str
        self.scale_factor: float
        self.a: float
        self.redshift: float
        self.z: float
        self.snapshot_path: str
        self.catalogue_properties_path: str

        for key in info_dict:
            setattr(self, key, info_dict[key])

        setattr(self, 'a', self.scale_factor)
        setattr(self, 'z', self.redshift)

    def __str__(self):
        return (
            f"Run name:                 {self.run_name}\n"
            f"Scale factor (a):         {self.scale_factor}\n"
            f"Redshift (z):             {self.redshift}\n"
            f"Snapshot file:            {self.snapshot_path}\n"
            f"Catalog properties file:  {self.catalogue_properties_path}"
        )


class Zoom(object):
    __slots__ = (
        'run_name',
        'run_directory',
        'redshifts',
        'scale_factors',
        'index_snaps',
        'index_snips',
        'snapshot_paths',
        'catalogue_properties_paths',
    )

    def __init__(self, run_directory: str) -> None:

        # Initialise variables as instance attributes, not as class attributes
        self.run_name: str
        self.run_directory: str
        self.redshifts: np.ndarray
        self.scale_factors: np.ndarray
        self.index_snaps: np.ndarray
        self.index_snips: np.ndarray
        self.snapshot_paths: List[str]
        self.catalogue_properties_paths: List[str]

        self.run_name = os.path.basename(run_directory)
        self.run_directory = run_directory
        self.redshifts, self.scale_factors, self.index_snaps, self.index_snips = self.read_output_list()

        # Retrieve absolute data paths to files
        snapshot_files = os.listdir(os.path.join(self.run_directory, 'snapshots'))
        snapshot_files = [file_name for file_name in snapshot_files if file_name.endswith('.hdf5')]

        # Sort filenames by snapshot file
        snapshot_files.sort(key=lambda x: int(x[-9:-5]))
        self.snapshot_paths = []
        self.catalogue_properties_paths = []

        for snap_path in snapshot_files:
            self.snapshot_paths.append(
                os.path.join(
                    self.run_directory,
                    'snapshots',
                    snap_path
                )
            )
            self.catalogue_properties_paths.append(
                os.path.join(
                    self.run_directory,
                    'stf',
                    os.path.splitext(snap_path)[0],
                    f"{os.path.splitext(snap_path)[0]}.properties"
                )
            )

        if xlargs.debug:

            assert len(self.redshifts) == len(self.scale_factors), (
                f"[Halo {self.run_name}] {len(self.redshifts)} != {len(self.scale_factors)}"
            )
            assert len(self.redshifts[self.index_snaps]) == len(self.snapshot_paths), (
                f"[Halo {self.run_name}] {len(self.redshifts[self.index_snaps])} != {len(self.snapshot_paths)}"
            )
            assert len(self.redshifts[self.index_snaps]) == len(self.catalogue_properties_paths), (
                f"[Halo {self.run_name}] {len(self.redshifts[self.index_snaps])} != {len(self.catalogue_properties_paths)}"
            )

    def filter_snaps_by_redshift(self, z_min: float = 0., z_max: float = 5., hard_check: bool = False):

        redshifts = self.redshifts[self.index_snaps]
        index_filter = np.where((redshifts > z_min) & (redshifts < z_max))[0]
        filtered_snapshot_paths = self.snapshot_paths[index_filter]
        filtered_catalogue_properties_paths = self.catalogue_properties_paths[index_filter]

        if hard_check:
            for file in filtered_snapshot_paths:
                with h5py.File(file, 'r') as f:
                    z = f['Header'].attrs['Redshift'][0]
                assert z_min < z < z_max

            for file in filtered_catalogue_properties_paths:
                with h5py.File(file, 'r') as f:
                    scale_factor = float(f['/SimulationInfo'].attrs['ScaleFactor'])
                    z = 1 / scale_factor - 1
                assert z_min < z < z_max

        return (
            filtered_snapshot_paths,
            filtered_catalogue_properties_paths,
            index_filter
        )

    def get_snip_handles(self, z_min: float = 0., z_max: float = 5.):

        assert len(self.index_snips) > 0, "No snipshots registered in the output list."
        snip_path = os.path.join(self.run_directory, 'snapshots')
        snip_handles = []

        with zipfile.ZipFile(os.path.join(snip_path, 'snipshots.zip'), 'r') as archive:
            all_snips = archive.namelist()
            all_snips.sort(key=lambda x: int(x[-9:-5]))
            for snip_name in tqdm(all_snips, desc=f"Fetching snipshots", disable=xlargs.quiet):
                snip_handle = io.BytesIO(archive.open(snip_name).read())
                with h5py.File(snip_handle, 'r') as f:
                    z = f['Header'].attrs['Redshift'][0]
                # Filter redshifts
                if z_min < z < z_max:
                    snip_handles.append(snip_handle)

        return snip_handles

    def read_output_list(self):
        output_list_file = os.path.join(self.run_directory, 'snap_redshifts.txt')
        output_list = pd.read_csv(output_list_file)

        # Need extra spaces because pandas doesn't seem to recognise the space after
        # a comma as a valid delimiter by default.
        redshifts = output_list["# Redshift"].values
        scale_factors = 1 / (redshifts + 1)

        if " Select Output" in output_list.columns:
            index_snaps = np.arange(len(output_list))[
                np.logical_or.reduce(
                    [output_list[" Select Output"] == f" Snapshot"]
                )
            ]

            index_snips = np.arange(len(output_list))[
                np.logical_or.reduce(
                    [output_list[" Select Output"] == f" Snipshot"]
                )
            ]

        else:
            index_snaps = np.arange(len(output_list))
            index_snips = np.empty(0, dtype=np.int)

        return redshifts, scale_factors, index_snaps, index_snips

    def get_redshift(self, index: int = -1):
        """
        To get z = 0 data promptly, specify index = -1. This
        selects the last output in the index list, which is the
        last redshift produced at runtime.

        :param index: int
            The integer index describing the output sequence.
        :return: Redshift instance
            The Redshift object contains fast-access absolute
            paths to the key files to read data from.
        """

        try:
            redshift_select = self.redshifts[self.index_snaps][index]
        except IndexError as err:
            print((
                f"Trying to access redshift with output index {index:d}, "
                f"but the maximum index available is {len(self.redshifts) - 1:d}."
            ))
            raise err

        redshift_info = dict()
        redshift_info['run_name'] = self.run_name
        redshift_info['scale_factor'] = self.scale_factors[self.index_snaps][index]
        redshift_info['redshift'] = redshift_select
        redshift_info['snapshot_path'] = self.snapshot_paths[index]
        redshift_info['catalogue_properties_path'] = self.catalogue_properties_paths[index]

        return Redshift(redshift_info)


def dump_exlzooms() -> None:
    calibration_zooms = EXLZooms()
    completed_runs = calibration_zooms.get_completed_run_directories()
    zooms_register = [Zoom(run_directory) for run_directory in completed_runs]
    name_list = calibration_zooms.get_completed_run_names()

    # Sort zooms by VR number
    zooms_register.sort(key=lambda x: int(x.run_name.split('_')[1][2:]))
    name_list.sort(key=lambda x: int(x.split('_')[1][2:]))

    pickler = MultiObjPickler('zooms_register.pkl', relative_path=True)
    pickler.dump_to_pickle(
        [
            calibration_zooms,
            completed_runs,
            zooms_register,
            name_list
        ]
    )


def load_exlzooms() -> list:
    pickler = MultiObjPickler('zooms_register.pkl', relative_path=True)
    try:
        data_pkl = pickler.load_from_pickle()
        return data_pkl
    except FileNotFoundError as e:
        print(
            "The EXL Zoom management data couldn't be found.",
            "Run the pipeline with the -refresh or --refresh-catalogue options.",
            e
        )


if xlargs.refresh:
    dump_exlzooms()

calibration_zooms, completed_runs, zooms_register, name_list = load_exlzooms()
