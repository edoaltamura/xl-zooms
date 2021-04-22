import sys
import os.path
import numpy as np
from copy import copy
from warnings import warn
from unyt import mp, cm, Mpc, unyt_quantity

from .halo_property import HaloProperty
from register import Zoom, Tcut_halogas, default_output_directory, args

sys.path.append("../xray")

import cloudy_softband as cloudy


class XrayLuminosities(HaloProperty):

    def __init__(self):
        super().__init__()

        self.labels = ['LX500', 'LX2500', 'LX500_nocore', 'LX2500_nocore']

        self.filename = os.path.join(
            default_output_directory,
            'intermediate',
            f'xray_luminosities_{args.redshift_index:04d}.pkl'
        )

    def check_value(self, value):

        if value >= 1:
            raise RuntimeError((
                f"The value for {self.labels[1]} must be between 0 and 1. "
                f"Got {value} instead."
            ))
        elif 0.5 < value < 1:
            warn(f"The value for {self.labels[1]} seems too high: {value}", RuntimeWarning)

    @staticmethod
    def get_emissivities(sw_data, mask):

        masked_sw_data = copy(sw_data)
        masked_sw_data.gas.densities = masked_sw_data.gas.densities[mask]
        masked_sw_data.gas.temperatures = masked_sw_data.gas.temperatures[mask]
        masked_sw_data.gas.masses = masked_sw_data.gas.masses[mask]
        masked_sw_data.gas.element_mass_fractions.hydrogen = masked_sw_data.gas.element_mass_fractions.hydrogen[mask]
        masked_sw_data.gas.element_mass_fractions.helium = masked_sw_data.gas.element_mass_fractions.helium[mask]
        masked_sw_data.gas.element_mass_fractions.carbon = masked_sw_data.gas.element_mass_fractions.carbon[mask]
        masked_sw_data.gas.element_mass_fractions.nitrogen = masked_sw_data.gas.element_mass_fractions.nitrogen[mask]
        masked_sw_data.gas.element_mass_fractions.oxygen = masked_sw_data.gas.element_mass_fractions.oxygen[mask]
        masked_sw_data.gas.element_mass_fractions.neon = masked_sw_data.gas.element_mass_fractions.neon[mask]
        masked_sw_data.gas.element_mass_fractions.magnesium = masked_sw_data.gas.element_mass_fractions.magnesium[mask]
        masked_sw_data.gas.element_mass_fractions.silicon = masked_sw_data.gas.element_mass_fractions.silicon[mask]
        masked_sw_data.gas.element_mass_fractions.iron = masked_sw_data.gas.element_mass_fractions.iron[mask]

        # Compute hydrogen number density and the log10
        # of the temperature to provide to the xray interpolator.
        data_nH = np.log10(
            masked_sw_data.gas.element_mass_fractions.hydrogen * masked_sw_data.gas.densities.to('g*cm**-3') / mp)
        data_T = np.log10(masked_sw_data.gas.temperatures.value)

        # Interpolate the Cloudy table to get emissivities
        emissivities = cloudy.interpolate_xray(
            data_nH,
            data_T,
            masked_sw_data.gas.element_mass_fractions
        )

        log10_Mpc3_to_cm3 = np.log10(Mpc.get_conversion_factor(cm)[0] ** 3)

        # The `data.gas.masses` and `data.gas.densities` datasets are formatted as
        # `numpy.float32` and are not well-behaved when trying to convert their ratio
        # from `unyt.Mpc ** 3` to `unyt.cm ** 3`, giving overflows and inf returns.
        # The `logsumexp` function offers a workaround to solve the problem when large
        # or small exponentiated numbers need to be summed (and logged) again.
        # See https://en.wikipedia.org/wiki/LogSumExp for details.
        # The conversion from `unyt.Mpc ** 3` to `unyt.cm ** 3` is also obtained by
        # adding the log10 of the conversion factor (2.9379989445851786e+73) to the
        # result of the `logsumexp` function.
        # $L_X = 10^{\log_{10} (\sum_i \epsilon_i) + log10_Mpc3_to_cm3}$
        LX = unyt_quantity(
            10 ** (
                    cloudy.logsumexp(
                        emissivities,
                        b=(masked_sw_data.gas.masses / masked_sw_data.gas.densities).value,
                        base=10.
                    ) + log10_Mpc3_to_cm3
            ), 'erg/s'
        )

        del masked_sw_data, data_nH, data_T, emissivities

        return LX

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
        LX500 = self.get_emissivities(sw_data, mask)

        mask = np.where(
            (sw_data.gas.radial_distances <= r2500) &
            (sw_data.gas.temperatures > Tcut_halogas) &
            (sw_data.gas.fofgroup_ids == 1)
        )[0]
        LX2500 = self.get_emissivities(sw_data, mask)

        # Select hot gas within spherical shell (no core)
        mask = np.where(
            (sw_data.gas.radial_distances >= 0.15 * r500) &
            (sw_data.gas.radial_distances <= r500) &
            (sw_data.gas.temperatures > Tcut_halogas) &
            (sw_data.gas.fofgroup_ids == 1)
        )[0]
        LX500_nocore = self.get_emissivities(sw_data, mask)

        mask = np.where(
            (sw_data.gas.radial_distances >= 0.15 * r500) &
            (sw_data.gas.radial_distances <= r2500) &
            (sw_data.gas.temperatures > Tcut_halogas) &
            (sw_data.gas.fofgroup_ids == 1)
        )[0]
        LX2500_nocore = self.get_emissivities(sw_data, mask)

        return LX500, LX2500, LX500_nocore, LX2500_nocore

    def process_catalogue(self):

        catalogue = self._process_catalogue(self.process_single_halo, labels=self.labels)
        self.dump_to_pickle(self.filename, catalogue)

    def read_catalogue(self):

        return self._read_catalogue(self.filename)

    def get_zoom_from_catalogue(self, **kwargs):

        return self._get_zoom_from_catalogue(self.filename, **kwargs)
