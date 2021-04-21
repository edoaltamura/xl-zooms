import os.path
import numpy as np
from warnings import warn

from .halo_property import HaloProperty
from register import Zoom, Tcut_halogas, default_output_directory, args


class GasFractions(HaloProperty):

    def __init__(self):
        super().__init__()

        self.labels = ['m_gas', 'f_gas']

        self.filename = os.path.join(
            default_output_directory,
            'intermediate',
            f'gas_fractions_{args.redshift_index:04d}.pkl'
        )

    def check_value(self, value):

        if value >= 1:
            raise RuntimeError((
                f"The value for {self.labels[0]} must be between 0 and 1. "
                f"Got {value} instead."
            ))
        elif 0.5 < value < 1:
            warn(f"The value for {self.labels[0]} seems too high: {value}", RuntimeWarning)

    def process_single_halo(
            self,
            zoom_obj: Zoom = None,
            path_to_snap: str = None,
            path_to_catalogue: str = None,
            **kwargs
    ):
        sw_data, vr_data = self.get_handles_from_zoom(zoom_obj, path_to_snap, path_to_catalogue, **kwargs)

        m500 = vr_data.spherical_overdensities.mass_500_rhocrit[0].to('Msun')
        r500 = vr_data.spherical_overdensities.r_500_rhocrit[0].to('Mpc')

        sw_data.gas.radial_distances.convert_to_physical()
        sw_data.gas.temperatures.convert_to_physical()

        # Select hot gas within sphere
        mask = np.where(
            (sw_data.gas.radial_distances <= r500) &
            (sw_data.gas.temperatures > Tcut_halogas) &
            (sw_data.gas.fofgroup_ids == 1)
        )[0]
        mhot500 = np.sum(sw_data.gas.masses[mask])
        mhot500 = mhot500.to('Msun')
        gas_fraction = mhot500 / m500

        self.check_value(gas_fraction)
        return mhot500, gas_fraction

    def process_catalogue(self):

        catalogue = self._process_catalogue(self.process_single_halo, labels=self.labels)
        self.dump_to_pickle(self.filename, catalogue)

    def read_catalogue(self):

        return self._read_catalogue(self.filename)

    def get_zoom_from_catalogue(self, **kwargs):

        return self._get_zoom_from_catalogue(self.filename, **kwargs)
