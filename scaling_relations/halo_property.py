import sys
import numpy as np
import swiftsimio
import velociraptor
from scipy.spatial import distance
import pandas as pd
from typing import Callable, List, Union
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor

sys.path.append("..")

from register import (
    args,
    DataframePickler,
    Zoom,
zooms_register
)


class HaloProperty(object):

    def __init__(self):
        pass

    @staticmethod
    def get_handles_from_paths(
            path_to_snap: str,
            path_to_catalogue: str,
            mask_radius_r500: float = 3
    ) -> tuple:
        """
        All quantities in the VR file are physical.
        All quantities in the Swiftsimio file are comoving. Convert them upon use.
        Args:
            path_to_snap:
            path_to_catalogue:
            mask_radius_r500:

        Returns:

        TODO:
            * Coordinate wrapping compatible with unyt_array
        """
        # Read in halo properties
        vr_handle = velociraptor.load(path_to_catalogue)
        a = vr_handle.a
        r500 = vr_handle.spherical_overdensities.r_500_rhocrit[0].to('Mpc') / a
        xcminpot = vr_handle.positions.xcminpot[0].to('Mpc') / a
        ycminpot = vr_handle.positions.ycminpot[0].to('Mpc') / a
        zcminpot = vr_handle.positions.zcminpot[0].to('Mpc') / a
        del a

        # Apply spatial mask to particles. SWIFTsimIO needs comoving coordinates
        # to filter particle coordinates, while VR outputs are in physical units.
        # Convert the region bounds to comoving, but keep the CoP and Rcrit in
        # physical units for later use.
        mask = swiftsimio.mask(path_to_snap)
        mask_radius = mask_radius_r500 * r500
        region = [
            [xcminpot - mask_radius, xcminpot + mask_radius],
            [ycminpot - mask_radius, ycminpot + mask_radius],
            [zcminpot - mask_radius, zcminpot + mask_radius]
        ]
        mask.constrain_spatial(region)
        sw_handle = swiftsimio.load(path_to_snap, mask=mask)

        if len(sw_handle.gas.coordinates) == 0:
            raise ValueError((
                "The spatial masking of the snapshot returned 0 particles. "
                "Check whether an appropriate aperture is selected and if "
                "the physical/comoving units match."
            ))
        else:
            print((
                "Loaded particles "
                f"[{len(sw_handle.gas.coordinates):d} gas] "
                f"[{len(sw_handle.dark_matter.coordinates):d} dark_matter] "
                f"[{len(sw_handle.stars.coordinates):d} stars] "
                f"[{len(sw_handle.black_holes.coordinates):d} black_holes] "
            ))

        # If the mask overlaps with the box boundaries, wrap coordinates.
        boxsize = sw_handle.metadata.boxsize[0]
        centre_coordinates = np.array([xcminpot, ycminpot, zcminpot], dtype=np.float64)

        # sw_handle.gas.coordinates = np.mod(
        #     sw_handle.gas.coordinates - centre_coordinates + 0.5 * boxsize,
        #     boxsize
        # ) + centre_coordinates - 0.5 * boxsize
        #
        # sw_handle.dark_matter.coordinates = np.mod(
        #     sw_handle.dark_matter.coordinates - centre_coordinates + 0.5 * boxsize,
        #     boxsize
        # ) + centre_coordinates - 0.5 * boxsize
        #
        # sw_handle.stars.coordinates = np.mod(
        #     sw_handle.stars.coordinates - centre_coordinates + 0.5 * boxsize,
        #     boxsize
        # ) + centre_coordinates - 0.5 * boxsize
        #
        # sw_handle.black_holes.coordinates = np.mod(
        #     sw_handle.black_holes.coordinates - centre_coordinates + 0.5 * boxsize,
        #     boxsize
        # ) + centre_coordinates - 0.5 * boxsize

        # Compute radial distances
        sw_handle.gas.radial_distances = swiftsimio.cosmo_array(
            distance.cdist(
                sw_handle.gas.coordinates,
                centre_coordinates.reshape(1, 3),
                metric='euclidean'
            ).reshape(len(sw_handle.gas.coordinates), ),
            units='Mpc',
            cosmo_factor=sw_handle.gas.coordinates.cosmo_factor
        )

        sw_handle.dark_matter.radial_distances = swiftsimio.cosmo_array(
            distance.cdist(
                sw_handle.dark_matter.coordinates,
                centre_coordinates.reshape(1, 3),
                metric='euclidean'
            ).reshape(len(sw_handle.dark_matter.coordinates), ),
            units='Mpc',
            cosmo_factor=sw_handle.dark_matter.coordinates.cosmo_factor
        )

        sw_handle.stars.radial_distances = swiftsimio.cosmo_array(
            distance.cdist(
                sw_handle.stars.coordinates,
                centre_coordinates.reshape(1, 3),
                metric='euclidean'
            ).reshape(len(sw_handle.stars.coordinates), ),
            units='Mpc',
            cosmo_factor=sw_handle.stars.coordinates.cosmo_factor
        )

        sw_handle.black_holes.radial_distances = swiftsimio.cosmo_array(
            distance.cdist(
                sw_handle.black_holes.coordinates,
                centre_coordinates.reshape(1, 3),
                metric='euclidean'
            ).reshape(len(sw_handle.black_holes.coordinates), ),
            units='Mpc',
            cosmo_factor=sw_handle.black_holes.coordinates.cosmo_factor
        )

        return sw_handle, vr_handle

    def get_handles_from_zoom(
            self,
            zoom_obj: Zoom = None,
            path_to_snap: str = None,
            path_to_catalogue: str = None,
            **kwargs
    ):
        assert (
                zoom_obj is not None or
                (path_to_snap is not None and path_to_snap is not None)
        ), (
            "Either a `Zoom` object must be specified or the absolute "
            "paths to the snapshot and properties catalogue files."
        )

        snapshot_file = path_to_snap
        catalog_file = path_to_catalogue

        if zoom_obj is not None:
            zoom_at_redshift = zoom_obj.get_redshift(args.redshift_index)
            snapshot_file = zoom_at_redshift.snapshot_path
            catalog_file = zoom_at_redshift.catalogue_properties_path

        return self.get_handles_from_paths(snapshot_file, catalog_file, **kwargs)

    def read_from_file(self):
        pickler = DataframePickler(self.storage_file)
        return pickler.load_from_pickle()


    def process_catalogue(self, _process_single_halo: Callable,
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

        @args find_keyword: specifies a keyword to filter the zooms register by.

        @args save_dataframe: set to True if you want to save the intermediate
        output of the calculation. Currently using pickle files generated by
        Pandas.
        """
        # Print the CLI arguments that are parsed in the script


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

        return results