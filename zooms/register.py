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

    SILENT_PROGRESSBAR (bool): Specifies whether progress-bars called with
    `tqdm` should appear in the console when the pipeline is run. E.g. for
    long computations, or when running in the background with `nohup [cmd] &`,
    you may want to set this to `True` to avoid printing the progress-bar.

    zooms_register (List[Zoom]): specifies the zooms catalogues formed with
    runs that have completed

Todo:
    * Finish developing the analyse_incomplete_runs method
    * Make Zoom attributes slots and static declarations


To maintain a readable and extensible documentation, please refer to this
guidelines:
.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import io
import os
import h5py
import psutil
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List

Tcut_halogas = 1.e5
SILENT_PROGRESSBAR = False


def dump_memory_usage() -> None:
    total_memory = psutil.virtual_memory().total
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss  # in bytes
    print((
        f"[Resources] Memory usage: {memory / 1024 / 1024:.2f} MB "
        f"({memory / total_memory * 100:.2f}%)"
    ))


class EXLZooms:
    name: str = 'Eagle-XL Zooms'
    output_dir: str = "/cosma7/data/dp004/dc-alta2/xl-zooms/analysis"

    # Zooms will be searched in this directories
    cosma_repositories: List[str] = [
        "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro",
        # "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro",
        # "/snap7/scratch/dp004/dc-alta2/xl-zooms/hydro",
    ]

    name_list: List[str] = []
    run_directories: List[str] = []
    complete_runs: np.ndarray

    def __init__(self) -> None:

        # Search for any run directories in repositories
        for repository in self.cosma_repositories:
            for run_basename in os.listdir(repository)[:15]:
                run_abspath = os.path.join(repository, run_basename)
                if os.path.isdir(run_abspath) and run_basename.startswith('L0300N0564'):
                    self.run_directories.append(run_abspath)
                    self.name_list.append(run_basename)

        # Classify complete and incomplete runs
        self.complete_runs = np.zeros(len(self.name_list), dtype=np.bool)

        for i, run_directory in enumerate(self.run_directories):
            snaps_path = os.path.join(run_directory, 'snapshots')
            catalogues_path = os.path.join(run_directory, 'stf')
            print(snaps_path)

            if os.path.isdir(snaps_path):
                number_snapshots = len([file for file in os.listdir(snaps_path) if file.endswith('.hdf5')])
            else:
                number_snapshots = 0

            if os.path.isdir(catalogues_path):
                number_catalogues = len([subdir for subdir in os.listdir(catalogues_path)])
            else:
                number_catalogues = 0

            if (
                    (number_snapshots > 0) and
                    (number_catalogues > 0) and
                    (number_snapshots == number_catalogues)
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

    def get_completed_catalogue(self):
        return [self.run_directories[i] for i in self.complete_runs if i]

    def get_incomplete_catalogue(self):
        return [self.run_directories[i] for i in self.complete_runs if not i]

    def analyse_incomplete_runs(self):

        incomplete_name_list = self.name_list[~self.complete_runs]
        incomplete_run_directories = self.run_directories[~self.complete_runs]

        for run_directory in incomplete_run_directories:
            for file in os.listdir(run_directory):
                if file.startswith('timesteps'):
                    timesteps_file = os.path.join(run_directory, file)

        with open(timesteps_file, 'r') as file_handle:
            lastlast_line = file_handle.readlines()[-2].split()
            last_line = file_handle.readlines()[-1].split()

            if len(lastlast_line) == len(last_line):
                last_redshift = float(last_line[3])
            elif len(lastlast_line) > len(last_line):
                last_redshift = float(lastlast_line[3])


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

    run_name: str
    scale_factor: float
    a: float
    redshift: float
    z: float
    snapshot_path: str
    catalogue_properties_path: str

    def __init__(self, info_dict: dict):
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

    def __init__(self, run_directory: str) -> None:
        self.run_name = os.path.basename(run_directory)
        self.run_directory = run_directory
        self.redshifts, self.scale_factors, self.index_snaps, self.index_snips = self.read_output_list()

        # Retrieve absolute data paths to files
        self.snapshot_paths = []
        self.catalogue_properties_paths = []

        snapshot_files = os.listdir(os.path.join(self.run_directory, 'snapshots'))
        snapshot_files = [file_name for file_name in snapshot_files if file_name.endswith('.hdf5')]

        # Sort filenames by snapshot file
        snapshot_files.sort(key=lambda x: int(x[-9:-5]))

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
            for snip_name in tqdm(all_snips, desc=f"Fetching snipshots", disable=SILENT_PROGRESSBAR):
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
        redshift_info['catalogue_properties_paths'] = self.catalogue_properties_paths[index]

        return Redshift(redshift_info)


calibration_zooms = EXLZooms()
completed_runs = calibration_zooms.get_completed_catalogue()
zooms_register = [Zoom(run_directory) for run_directory in completed_runs]

# Sort zooms by VR number
zooms_register.sort(key=lambda x: int(x.run_name.split('_')[1][2:]))

if __name__ == "__main__":
    incomplete_runs = calibration_zooms.get_incomplete_catalogue()
    print((
        "\n"
        "The following simulations were found with directory set-up, "
        "but missing snapshots or stf sub-directories. They are "
        "likely not yet launched or incomplete and were not appended "
        "to the master register."
    ))
    for i in incomplete_runs:
        print(f"[!] -> {i:s}")

    print(f"\n{' Zoom register ':-^40s}")
    for i in completed_runs:
        print(i)
