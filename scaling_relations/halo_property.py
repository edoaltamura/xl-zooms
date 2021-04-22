import sys
import numpy as np
import swiftsimio
import velociraptor
from scipy.spatial import distance
import pandas as pd
from tqdm import tqdm
from unyt import unyt_array
from typing import Tuple, List
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor

sys.path.append("..")

from register import (
    args,
    DataframePickler,
    Zoom,
    zooms_register,
    calibration_zooms
)


def histogram_unyt(data: unyt_array,
                   bins: unyt_array = None,
                   weights: unyt_array = None) -> Tuple[unyt_array]:
    assert data.shape == weights.shape, (
        "Data and weights arrays must have the same shape. "
        f"Detected data {data.shape}, weights {weights.shape}."
    )

    assert data.units == bins.units, (
        "Data and bins must have the same units. "
        f"Detected data {data.units}, bins {bins.units}."
    )

    hist, bin_edges = np.histogram(data.value, bins=bins.value, weights=weights.value)
    hist *= weights.units
    bin_edges *= data.units

    return hist, bin_edges


def cumsum_unyt(data: unyt_array) -> unyt_array:
    res = np.cumsum(data.value)

    return res * data.units


class HaloProperty(object):

    def __init__(self):
        pass

    @staticmethod
    def wrap_coordinates(
            coords: swiftsimio.cosmo_array,
            centre: unyt_array,
            boxsize: unyt_array
    ) -> swiftsimio.cosmo_array:
        result_numeric = np.mod(
            coords.value - centre.value + 0.5 * boxsize.value,
            boxsize.value
        ) + centre.value - 0.5 * boxsize.value

        result = swiftsimio.cosmo_array(
            result_numeric,
            units=coords.units,
            cosmo_factor=coords.cosmo_factor
        )

        return result

    @staticmethod
    def get_radial_distance(
            coords: swiftsimio.cosmo_array,
            centre: unyt_array
    ) -> swiftsimio.cosmo_array:
        result = swiftsimio.cosmo_array(
            distance.cdist(
                coords,
                centre.reshape(1, 3),
                metric='euclidean'
            ).reshape(len(coords), ),
            units='Mpc',
            cosmo_factor=coords.cosmo_factor
        )

        return result

    def get_handles_from_paths(
            self,
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
        # else:
        #     print((
        #         "Loaded particles "
        #         f"[{len(sw_handle.gas.coordinates):d} gas] "
        #         f"[{len(sw_handle.dark_matter.coordinates):d} dark_matter] "
        #         f"[{len(sw_handle.stars.coordinates):d} stars] "
        #         f"[{len(sw_handle.black_holes.coordinates):d} black_holes] "
        #     ))

        # If the mask overlaps with the box boundaries, wrap coordinates.
        boxsize = sw_handle.metadata.boxsize
        centre_coordinates = unyt_array([xcminpot, ycminpot, zcminpot], xcminpot.units)

        sw_handle.gas.coordinates = self.wrap_coordinates(
            sw_handle.gas.coordinates,
            centre_coordinates,
            boxsize
        )

        sw_handle.dark_matter.coordinates = self.wrap_coordinates(
            sw_handle.dark_matter.coordinates,
            centre_coordinates,
            boxsize
        )

        sw_handle.stars.coordinates = self.wrap_coordinates(
            sw_handle.stars.coordinates,
            centre_coordinates,
            boxsize
        )

        sw_handle.black_holes.coordinates = self.wrap_coordinates(
            sw_handle.black_holes.coordinates,
            centre_coordinates,
            boxsize
        )

        # Compute radial distances
        sw_handle.gas.radial_distances = self.get_radial_distance(
            sw_handle.gas.coordinates,
            centre_coordinates
        )

        sw_handle.dark_matter.radial_distances = self.get_radial_distance(
            sw_handle.dark_matter.coordinates,
            centre_coordinates
        )

        sw_handle.stars.radial_distances = self.get_radial_distance(
            sw_handle.stars.coordinates,
            centre_coordinates
        )

        sw_handle.black_holes.radial_distances = self.get_radial_distance(
            sw_handle.black_holes.coordinates,
            centre_coordinates
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

    @staticmethod
    def dump_to_pickle(storage_file, obj: pd.DataFrame):
        pickler = DataframePickler(storage_file)
        return pickler.dump_to_pickle(obj)

    @staticmethod
    def _read_catalogue(storage_file):
        pickler = DataframePickler(storage_file)
        return pickler.load_from_pickle()

    def _get_zoom_from_catalogue(self,
                                 storage_file: str,
                                 zoom_obj: Zoom = None,
                                 zoom_name: str = None) -> pd.DataFrame:

        assert zoom_obj is not None or zoom_name is not None, (
            "Need to specify either `zoom_obj` or `zoom_name`."
        )
        name = zoom_name
        if zoom_obj is not None:
            name = zoom_obj.run_name

        catalogue = self._read_catalogue(storage_file)

        if name not in catalogue['Run_name'].unique():
            raise RuntimeError((
                f"The {name} zoom could not be found in the catalogue "
                f"{storage_file}. Double check manually and regenerate "
                "the catalogue if necessary. The `--restart` option can "
                "be used."
            ))

        return catalogue.loc[catalogue['Run_name'] == name]

    @staticmethod
    def _process_catalogue(single_halo_method,
                           labels: List[str],
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

        if args.refresh:
            _zooms_register = zooms_register
        else:
            _zooms_register = []
            for keyword in args.keywords:
                for zoom in zooms_register:
                    if keyword in zoom.run_name and zoom not in _zooms_register:
                        _zooms_register.append(zoom)

        _name_list = [zoom.run_name for zoom in _zooms_register]

        if len(_zooms_register) == 1:
            print("Analysing one object only. Not using multiprocessing features.")
            results = [single_halo_method(_zooms_register[0])]

        else:

            if no_multithreading:
                print(f"Running with no multithreading.\nAnalysing {len(_zooms_register):d} zooms serially.")
                results = []
                for i, zoom in enumerate(_zooms_register):
                    print(f"({i + 1}/{len(_zooms_register)}) Processing: {zoom.run_name}")
                    results.append(
                        single_halo_method(zoom)
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
                        results = tqdm(
                            list(pool.imap(single_halo_method, iter(_zooms_register))),
                            total=len(_zooms_register)
                        )
                except Exception as error:
                    print((
                        f"The analysis stopped due to the error\n{error}\n"
                        "Please use a different multiprocessing pool or run the code serially."
                    ))
                    raise error

        # Recast output into a Pandas dataframe for further manipulation
        results = pd.DataFrame(list(results), columns=labels)
        results.insert(0, 'Run_name', pd.Series(_name_list, dtype=str))
        if not args.quiet:
            print(
                f"z = {calibration_zooms.redshift_from_index(args.redshift_index):.2f}"
                "\nOutput dataframe (head only):\n",
                results.head()
            )

        return results
