import os.path
import numpy as np
from typing import Union, Optional
from unyt import kb, mh, Mpc, dimensionless
from collections import namedtuple
from swiftsimio.visualisation.projection import project_pixel_grid
from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths
from swiftsimio.visualisation.slice import kernel_gamma


from .halo_property import HaloProperty
from register import Zoom, default_output_directory, xlargs


class MapDM(HaloProperty):

    def __init__(
            self,
            resolution: int = 1024,
            parallel: bool = True,
            backend: str = 'fast',
    ):
        super().__init__()

        self.labels = ['dm_map', 'region']

        self.resolution = resolution
        self.parallel = parallel
        self.backend = backend
        self.map_centre = 'vr_centre_of_potential'

        self.filename = os.path.join(
            default_output_directory,
            'intermediate',
            f'dm_map_{xlargs.redshift_index:04d}.pkl'
        )

    def process_single_halo(
            self,
            zoom_obj: Zoom = None,
            path_to_snap: str = None,
            path_to_catalogue: str = None,
            mask_radius_r500: float = 6,
            map_centre: Union[str, list, np.ndarray] = 'vr_centre_of_potential',
            depth: Optional[float] = None,
            return_type: Union[type, str] = 'class'
    ):
        sw_data, vr_data = self.get_handles_from_zoom(
            zoom_obj,
            path_to_snap,
            path_to_catalogue,
            mask_radius_r500=mask_radius_r500,
        )

        map_centres_allowed = [
            'vr_centre_of_potential'
        ]

        if type(map_centre) is str and map_centre.lower() not in map_centres_allowed:
            raise AttributeError((
                f"String-commands for `map_centre` only support "
                f"`vr_centre_of_potential`. Got {map_centre} instead."
            ))
        elif (type(map_centre) is list or type(map_centre) is np.ndarray) and len(map_centre) != 3:
            raise AttributeError((
                f"List-commands for `map_centre` only support "
                f"length-3 lists. Got {map_centre} "
                f"(length {len(map_centre)}) instead."
            ))

        self.map_centre = map_centre

        centre_of_potential = [
            vr_data.positions.xcminpot[0].to('Mpc') / vr_data.a,
            vr_data.positions.ycminpot[0].to('Mpc') / vr_data.a,
            vr_data.positions.zcminpot[0].to('Mpc') / vr_data.a
        ]

        if self.map_centre == 'vr_centre_of_potential':
            _xCen = vr_data.positions.xcminpot[0].to('Mpc') / vr_data.a
            _yCen = vr_data.positions.ycminpot[0].to('Mpc') / vr_data.a
            _zCen = vr_data.positions.zcminpot[0].to('Mpc') / vr_data.a

        elif type(self.map_centre) is list or type(self.map_centre) is np.ndarray:
            _xCen = self.map_centre[0] * Mpc / vr_data.a
            _yCen = self.map_centre[1] * Mpc / vr_data.a
            _zCen = self.map_centre[2] * Mpc / vr_data.a

        if xlargs.debug:
            print(f"Centre of potential: {[float(f'{i.v:.3f}') for i in centre_of_potential]} Mpc")
            print(f"Map centre: {[float(f'{i.v:.3f}') for i in [_xCen, _yCen, _zCen]]} Mpc")

        _r500 = vr_data.spherical_overdensities.r_500_rhocrit[0].to('Mpc') / vr_data.a

        region = [
            _xCen - mask_radius_r500 / np.sqrt(3) * _r500,
            _xCen + mask_radius_r500 / np.sqrt(3) * _r500,
            _yCen - mask_radius_r500 / np.sqrt(3) * _r500,
            _yCen + mask_radius_r500 / np.sqrt(3) * _r500
        ]

        if not hasattr(sw_data.dark_matter, 'smoothing_lengths'):
            print('Generate smoothing lengths for the dark matter')
            sw_data.dark_matter.smoothing_lengths = generate_smoothing_lengths(
                sw_data.dark_matter.coordinates,
                sw_data.metadata.boxsize,
                kernel_gamma=kernel_gamma * 0.7,
                neighbours=57,
                speedup_fac=2,
                dimension=3,
            )

        if depth is not None:

            depth = min(depth * Mpc, mask_radius_r500 * np.sqrt(3) * _r500)

            depth_filter = np.where(
                (sw_data.dark_matter.coordinates[:, -1] > _zCen - depth / 2) &
                (sw_data.dark_matter.coordinates[:, -1] < _zCen + depth / 2)
            )[0]

            if xlargs.debug:
                percent = f"{len(depth_filter) / len(sw_data.dark_matter.coordinates) * 100:.1f}"
                print((
                    f"Filtering particles by depth: +/- {depth:.2f}/2  Mpc.\n"
                    f"Total particles: {len(sw_data.dark_matter.coordinates)}\n"
                    f"Particles within bounds: {len(depth_filter)} = {percent} %"
                ))

            sw_data.dark_matter.coordinates = sw_data.dark_matter.coordinates[depth_filter]
            sw_data.dark_matter.smoothing_lengths = sw_data.dark_matter.smoothing_lengths[depth_filter]
            sw_data.dark_matter.masses = sw_data.dark_matter.masses[depth_filter]

        # Note here that we pass in the dark matter dataset not the whole
        # data object, to specify what particle type we wish to visualise
        dm_map = project_pixel_grid(
            project="masses",
            data=sw_data.dark_matter,
            resolution=self.resolution,
            parallel=self.parallel,
            region=region,
            backend=self.backend,
            boxsize=sw_data.metadata.boxsize
        )

        dm_map = np.ma.array(
            dm_map,
            mask=(dm_map <= 0.),
            fill_value=np.nan,
            copy=True,
            dtype=np.float64
        )

        output_values = [
            dm_map,
            region,
            dimensionless / Mpc ** 2,
            [_xCen, _yCen, _zCen],
            _r500,
            sw_data.metadata.z
        ]
        output_names = [
            'map',
            'region',
            'units',
            'centre',
            'r500',
            'z'
        ]
        if return_type is tuple:
            output = tuple(output_values)
        elif return_type is dict:
            output = dict(zip(output_names, output_values))
        elif return_type == 'class':
            OutputClass = namedtuple('OutputClass', output_names)
            output = OutputClass(*output_values)
        else:
            raise TypeError(f"Return type {return_type} not recognised.")

        return output

    def process_catalogue(self):

        catalogue = self._process_catalogue(self.process_single_halo, labels=self.labels)
        self.dump_to_pickle(self.filename, catalogue)

    def read_catalogue(self):

        return self._read_catalogue(self.filename)

    def get_zoom_from_catalogue(self, **kwargs):

        return self._get_zoom_from_catalogue(self.filename, **kwargs)
