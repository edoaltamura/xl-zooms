import os.path
import numpy as np
from warnings import warn

from .halo_property import HaloProperty
from register import Zoom, Tcut_halogas, default_output_directory, xlargs


class MWTemperatures(HaloProperty):

    def __init__(self):
        super().__init__()

        self.labels = ['T500', 'T2500', 'T500_nocore', 'T2500_nocore']

        self.filename = os.path.join(
            default_output_directory,
            'intermediate',
            f'mw_temperatures_{xlargs.mass_estimator:s}_{xlargs.redshift_index:04d}.pkl'
        )

    def check_value(self, value):

        if value >= 1:
            raise RuntimeError((
                f"The value for {self.labels[1]} must be between 0 and 1. "
                f"Got {value} instead."
            ))
        elif 0.5 < value < 1:
            warn(f"The value for {self.labels[1]} seems too high: {value}", RuntimeWarning)

    def process_single_halo(
            self,
            zoom_obj: Zoom = None,
            path_to_snap: str = None,
            path_to_catalogue: str = None,
            **kwargs
    ):
        sw_data, vr_data = self.get_handles_from_zoom(zoom_obj, path_to_snap, path_to_catalogue, **kwargs)

        r500 = vr_data.spherical_overdensities.r_500_rhocrit[0].to('Mpc')
        r2500 = vr_data.spherical_overdensities.r_2500_rhocrit[0].to('Mpc')

        sw_data.gas.radial_distances.convert_to_physical()
        sw_data.gas.temperatures.convert_to_physical()
        sw_data.gas.masses.convert_to_physical()

        sw_data.gas.mw_temperatures = sw_data.gas.temperatures * sw_data.gas.masses

        # Select hot gas within sphere
        mask = np.where(
            (sw_data.gas.radial_distances <= r500) &
            (sw_data.gas.temperatures > Tcut_halogas) &
            (sw_data.gas.fofgroup_ids == 1)
        )[0]
        mhot500 = np.sum(sw_data.gas.masses[mask])
        T500 = np.sum(sw_data.gas.mw_temperatures[mask]) / mhot500

        mask = np.where(
            (sw_data.gas.radial_distances <= r2500) &
            (sw_data.gas.temperatures > Tcut_halogas) &
            (sw_data.gas.fofgroup_ids == 1)
        )[0]
        mhot2500 = np.sum(sw_data.gas.masses[mask])
        T2500 = np.sum(sw_data.gas.mw_temperatures[mask]) / mhot2500

        # Select hot gas within spherical shell (no core)
        mask = np.where(
            (sw_data.gas.radial_distances >= 0.15 * r500) &
            (sw_data.gas.radial_distances <= r500) &
            (sw_data.gas.temperatures > Tcut_halogas) &
            (sw_data.gas.fofgroup_ids == 1)
        )[0]
        mhot500_nocore = np.sum(sw_data.gas.masses[mask])
        T500_nocore = np.sum(sw_data.gas.mw_temperatures[mask]) / mhot500_nocore

        mask = np.where(
            (sw_data.gas.radial_distances >= 0.15 * r500) &
            (sw_data.gas.radial_distances <= r2500) &
            (sw_data.gas.temperatures > Tcut_halogas) &
            (sw_data.gas.fofgroup_ids == 1)
        )[0]
        mhot2500_nocore = np.sum(sw_data.gas.masses[mask])
        T2500_nocore = np.sum(sw_data.gas.mw_temperatures[mask]) / mhot2500_nocore

        return T500, T2500, T500_nocore, T2500_nocore

    def process_catalogue(self):

        catalogue = self._process_catalogue(self.process_single_halo, labels=self.labels)
        self.dump_to_pickle(self.filename, catalogue)

    def read_catalogue(self):

        return self._read_catalogue(self.filename)

    def get_zoom_from_catalogue(self, **kwargs):

        return self._get_zoom_from_catalogue(self.filename, **kwargs)
