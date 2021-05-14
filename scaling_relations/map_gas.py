import os.path
import numpy as np
from warnings import warn
from typing import Union
from unyt import kb, mh
from swiftsimio.visualisation.projection import project_gas

from .halo_property import HaloProperty
from register import Zoom, default_output_directory, args

mean_molecular_weight = 0.59


class MapGas(HaloProperty):

    def __init__(
            self,
            project_quantity: str,
            resolution: int = 1024,
            parallel: bool = True,
            backend: str = 'fast',
    ):
        super().__init__()

        self.labels = ['gas_map', 'region']

        self.project_quantity = project_quantity
        self.resolution = resolution
        self.parallel = parallel
        self.backend = backend
        self.map_centre = 'vr_centre_of_potential'

        self.filename = os.path.join(
            default_output_directory,
            'intermediate',
            f'gas_map_{project_quantity:s}_{args.redshift_index:04d}.pkl'
        )

    def process_single_halo(
            self,
            zoom_obj: Zoom = None,
            path_to_snap: str = None,
            path_to_catalogue: str = None,
            mask_radius_r500: float = 6,
            map_centre: Union[str, list] = 'vr_centre_of_potential',
            **kwargs
    ):
        sw_data, vr_data = self.get_handles_from_zoom(
            zoom_obj,
            path_to_snap,
            path_to_catalogue,
            mask_radius_r500=mask_radius_r500,
            **kwargs
        )

        map_centres_allowed = [
            'vr_centre_of_potential'
        ]

        if type(map_centre) is str and map_centre.lower() not in map_centres_allowed:
            raise AttributeError((
                f"String-commands for `map_centre` only support "
                f"`vr_centre_of_potential`. Got {map_centre} instead."
            ))
        elif type(map_centre) is list and len(map_centre) not in [2, 3]:
            raise AttributeError((
                f"List-commands for `map_centre` only support "
                f"length-2 and length-3 lists. Got {map_centre} "
                f"(length {len(map_centre)}) instead."
            ))

        self.map_centre = map_centre

        if self.map_centre == 'vr_centre_of_potential':
            _xCen = vr_data.positions.xcminpot[0].to('Mpc') / vr_data.a
            _yCen = vr_data.positions.ycminpot[0].to('Mpc') / vr_data.a
        elif type(self.map_centre) is list:
            _xCen = self.map_centre[0]
            _yCen = self.map_centre[1]

        _r500 = vr_data.spherical_overdensities.r_500_rhocrit[0].to('Mpc') / vr_data.a
        region = [
            _xCen - mask_radius_r500 / np.sqrt(2) * _r500,
            _xCen + mask_radius_r500 / np.sqrt(2) * _r500,
            _yCen - mask_radius_r500 / np.sqrt(2) * _r500,
            _yCen + mask_radius_r500 / np.sqrt(2) * _r500
        ]

        if self.project_quantity == 'entropies':
            number_density = (sw_data.gas.densities / mh).to('cm**-3') / mean_molecular_weight
            entropy = kb * sw_data.gas.temperatures / number_density ** (2 / 3)
            sw_data.gas.entropies_physical = entropy.to('keV*cm**2')

            gas_map = project_gas(
                project='entropies_physical',
                data=sw_data,
                resolution=self.resolution,
                parallel=self.parallel,
                region=region,
                backend=self.backend
            ).value

        elif self.project_quantity == 'temperatures':
            sw_data.gas.mwtemps = sw_data.gas.masses * sw_data.gas.temperatures

            mass_weighted_temp_map = project_gas(
                project='mwtemps',
                data=sw_data,
                resolution=self.resolution,
                parallel=self.parallel,
                region=region,
                backend=self.backend
            ).value
            mass_map = project_gas(
                project='masses',
                data=sw_data,
                resolution=self.resolution,
                parallel=self.parallel,
                region=region,
                backend=self.backend
            ).value

            mass_weighted_temp_map = np.ma.array(
                mass_weighted_temp_map,
                mask=(mass_weighted_temp_map <= 0.),
                fill_value=np.nan,
                copy=True,
                dtype=np.float64
            )
            mass_map = np.ma.array(
                mass_map,
                mask=(mass_map <= 0.),
                fill_value=np.nan,
                copy=True,
                dtype=np.float64
            )
            gas_map = mass_weighted_temp_map / mass_map

        else:
            gas_map = project_gas(
                project=self.project_quantity,
                data=sw_data,
                resolution=self.resolution,
                parallel=self.parallel,
                region=region,
                backend=self.backend
            ).value
            gas_map = np.ma.array(
                gas_map,
                mask=(gas_map <= 0.),
                fill_value=np.nan,
                copy=True,
                dtype=np.float64
            )

        return gas_map, region

    def process_catalogue(self):

        catalogue = self._process_catalogue(self.process_single_halo, labels=self.labels)
        self.dump_to_pickle(self.filename, catalogue)

    def read_catalogue(self):

        return self._read_catalogue(self.filename)

    def get_zoom_from_catalogue(self, **kwargs):

        return self._get_zoom_from_catalogue(self.filename, **kwargs)
