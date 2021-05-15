import os.path
import numpy as np
from typing import Union, Optional
from unyt import kb, mh, Mpc, K
from swiftsimio.visualisation.slice import slice_gas

from .halo_property import HaloProperty
from register import Zoom, default_output_directory, args

mean_molecular_weight = 0.59


class SliceGas(HaloProperty):

    def __init__(
            self,
            project_quantity: str,
            resolution: int = 1024,
            parallel: bool = True,
    ):
        super().__init__()

        self.labels = ['gas_slice', 'region']

        self.project_quantity = project_quantity
        self.resolution = resolution
        self.parallel = parallel
        self.depth = 0
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
            temperature_range: Optional[tuple] = None,
            depth_offset: Optional[float] = None
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
        elif type(map_centre) is list and len(map_centre) != 3:
            raise AttributeError((
                f"List-commands for `map_centre` only support "
                f"length-3 lists. Got {map_centre} "
                f"(length {len(map_centre)}) instead."
            ))

        self.map_centre = map_centre

        if self.map_centre == 'vr_centre_of_potential':
            _xCen = vr_data.positions.xcminpot[0].to('Mpc') / vr_data.a
            _yCen = vr_data.positions.ycminpot[0].to('Mpc') / vr_data.a
            _zCen = vr_data.positions.zcminpot[0].to('Mpc') / vr_data.a

        elif type(self.map_centre[1]) is list:
            _xCen = self.map_centre[0]
            _yCen = self.map_centre[1]
            _zCen = self.map_centre[2]

        self.depth = _zCen / sw_data.metadata.boxsize[0]

        if depth_offset is not None:
            self.depth += depth_offset * Mpc / sw_data.metadata.boxsize[0]

            if args.debug:
                percent = f"{depth_offset * Mpc / _zCen * 100:.1f}"
                print((
                    f"Imposing offset in slicing depth: {depth_offset:.2f} Mpc.\n"
                    f"Percentage shift compared to centre: {percent} %"
                ))

        _r500 = vr_data.spherical_overdensities.r_500_rhocrit[0].to('Mpc') / vr_data.a
        region = [
            _xCen - mask_radius_r500 / np.sqrt(3) * _r500,
            _xCen + mask_radius_r500 / np.sqrt(3) * _r500,
            _yCen - mask_radius_r500 / np.sqrt(3) * _r500,
            _yCen + mask_radius_r500 / np.sqrt(3) * _r500
        ]

        if temperature_range is not None:

            temp_filter = np.where(
                (sw_data.gas.temperatures > temperature_range[0]) &
                (sw_data.gas.temperatures < temperature_range[1])
            )[0]

            if args.debug:
                percent = f"{len(temp_filter) / len(sw_data.gas.temperatures) * 100:.1f}"
                print((
                    f"Filtering particles by temperature: {temperature_range} K.\n"
                    f"Total particles: {len(sw_data.gas.temperatures)}\n"
                    f"Particles within bounds: {len(temp_filter)} = {percent} %"
                ))

            sw_data.gas.coordinates = sw_data.gas.coordinates[temp_filter]
            sw_data.gas.smoothing_lengths = sw_data.gas.smoothing_lengths[temp_filter]
            sw_data.gas.masses = sw_data.gas.masses[temp_filter]
            sw_data.gas.densities = sw_data.gas.densities[temp_filter]
            sw_data.gas.temperatures = sw_data.gas.temperatures[temp_filter]

        if self.project_quantity == 'entropies':
            number_density = (sw_data.gas.densities / mh).to('cm**-3') / mean_molecular_weight
            entropy = kb * sw_data.gas.temperatures / number_density ** (2 / 3)
            sw_data.gas.entropies_physical = entropy.to('keV*cm**2')

            gas_map = slice_gas(
                project='entropies_physical',
                data=sw_data,
                resolution=self.resolution,
                parallel=self.parallel,
                region=region,
                slice=self.depth
            ).value

        elif self.project_quantity == 'temperatures':
            sw_data.gas.mwtemps = sw_data.gas.masses * sw_data.gas.temperatures

            mass_weighted_temp_map = slice_gas(
                project='mwtemps',
                data=sw_data,
                resolution=self.resolution,
                parallel=self.parallel,
                region=region,
                slice=self.depth
            )
            mass_map = slice_gas(
                project='masses',
                data=sw_data,
                resolution=self.resolution,
                parallel=self.parallel,
                region=region,
                slice=self.depth
            )

            with np.errstate(divide='ignore', invalid='ignore'):
                gas_map = mass_weighted_temp_map / mass_map

            gas_map = gas_map.to(K).value

        else:
            gas_map = slice_gas(
                project=self.project_quantity,
                data=sw_data,
                resolution=self.resolution,
                parallel=self.parallel,
                region=region,
                slice=self.depth
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