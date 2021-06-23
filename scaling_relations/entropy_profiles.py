import os.path
import sys
import numpy as np
from warnings import warn
from matplotlib import pyplot as plt
import numba
from multiprocessing import cpu_count

numba.config.NUMBA_NUM_THREADS = cpu_count()

from unyt import (
    unyt_array,
    unyt_quantity,
    mh, G, mp, K, kb, cm, Solar_Mass, Mpc, dimensionless
)

from register import (
    Zoom, Tcut_halogas, default_output_directory, xlargs,
    set_mnras_stylesheet,
    mean_molecular_weight,
    mean_atomic_weight_per_free_electron,
    primordial_hydrogen_mass_fraction,
    solar_metallicity,
    gamma,
)
from .halo_property import HaloProperty, histogram_unyt
from .spherical_overdensities import SODelta500
from hydrostatic_estimates import HydrostaticEstimator
from .electron_number_density import get_electron_number_density, get_electron_number_density_shell_average
from literature import Cosmology, Sun2009, Pratt2010

sys.path.append("../xray")
import cloudy_softband as cloudy


class EntropyProfiles(HaloProperty):

    def __init__(
            self,
            max_radius_r500: float = 4,
            weighting: str = 'mass',
            simple_electron_number_density: bool = False,
            shell_average: bool = True
    ):
        super().__init__()

        self.labels = ['r', 'k', 'k500']
        self.max_radius_r500 = max_radius_r500
        self.weighting = weighting
        self.simple_electron_number_density = simple_electron_number_density
        self.shell_average = shell_average

        self.filename = os.path.join(
            default_output_directory,
            'intermediate',
            f'gas_fractions_{xlargs.mass_estimator:s}_{xlargs.redshift_index:04d}.pkl'
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
            path_to_catalogue: str = None
    ):

        sw_data, vr_data = self.get_handles_from_zoom(
            zoom_obj,
            path_to_snap,
            path_to_catalogue,
            mask_radius_r500=self.max_radius_r500
        )
        fb = Cosmology().fb0
        critical_density = unyt_quantity(
            sw_data.metadata.cosmology.critical_density(sw_data.metadata.z).value, 'g/cm**3'
        ).to('Msun/Mpc**3')

        sw_data.gas.radial_distances.convert_to_physical()
        sw_data.gas.masses.convert_to_physical()
        sw_data.gas.densities.convert_to_physical()
        sw_data.gas.entropies.convert_to_physical()

        try:
            m500 = vr_data.spherical_overdensities.mass_500_rhocrit[0].to('Msun')
            r500 = vr_data.spherical_overdensities.r_500_rhocrit[0].to('Mpc')
        except AttributeError as err:
            print(f'[{self.__class__.__name__}] {err}')

            spherical_overdensity = SODelta500(
                path_to_snap=path_to_snap,
                path_to_catalogue=path_to_catalogue,
            )
            m500 = spherical_overdensity.get_m500()
            r500 = spherical_overdensity.get_r500()

        if xlargs.mass_estimator == 'hse':
            true_hse = HydrostaticEstimator(
                path_to_catalogue=path_to_catalogue,
                path_to_snap=path_to_snap,
                profile_type='true',
                diagnostics_on=False
            )
            true_hse.interpolate_hse(density_contrast=500.)
            r500 = true_hse.r500hse
            m500 = true_hse.m500hse

        try:
            _ = sw_data.gas.temperatures
        except AttributeError as err:
            print(f'[{self.__class__.__name__}] {err}')
            if xlargs.debug:
                print(f"[{self.__class__.__name__}] Computing gas temperature from internal energies.")
            sw_data.gas.temperatures = sw_data.gas.internal_energies * (gamma - 1) * mean_molecular_weight * mh / kb

        try:
            _ = sw_data.gas.fofgroup_ids
        except AttributeError as err:
            print(f'[{self.__class__.__name__}] {err}')
            if xlargs.debug:
                print(f"[{self.__class__.__name__}] Select particles only by radial distance.")
            sw_data.gas.fofgroup_ids = np.ones_like(sw_data.gas.densities)

        index = np.where(
            (sw_data.gas.radial_distances < self.max_radius_r500 * r500) &
            (sw_data.gas.fofgroup_ids == 1) &
            (sw_data.gas.temperatures > Tcut_halogas)
        )[0]
        radial_distance = sw_data.gas.radial_distances[index] / r500

        # Define radial bins and shell volumes
        lbins = np.logspace(-2, np.log10(self.max_radius_r500), 51) * radial_distance.units
        radial_bin_centres = 10 ** (0.5 * np.log10(lbins[1:] * lbins[:-1])) * radial_distance.units

        if self.weighting == 'xray':
            # Compute hydrogen number density and the log10
            # of the temperature to provide to the xray interpolator.
            data_nH = np.log10(
                sw_data.gas.element_mass_fractions.hydrogen * sw_data.gas.densities.to('g*cm**-3') / mp)
            data_T = np.log10(sw_data.gas.temperatures.value)

            # Interpolate the Cloudy table to get emissivities
            emissivities = unyt_array(
                10 ** cloudy.interpolate_X_Ray(
                    data_nH,
                    data_T,
                    sw_data.gas.element_mass_fractions,
                    fill_value=-50.
                ), 'erg/s/cm**3'
            )
            xray_luminosities = emissivities * sw_data.gas.masses / sw_data.gas.densities
            weighting = xray_luminosities[index]
            del data_nH, data_T, emissivities, xray_luminosities

        elif self.weighting == 'mass':
            weighting = sw_data.gas.masses[index]

        elif self.weighting == 'volume':
            volume_proxy = sw_data.gas.masses[index] / sw_data.gas.densities[index]
            weighting = volume_proxy
            del volume_proxy

        if self.simple_electron_number_density:
            sw_data.gas.densities.convert_to_units('g/cm**3')
            n_e = sw_data.gas.densities[index] / (mp * mean_atomic_weight_per_free_electron)
            n_e.convert_to_units('cm**-3')
        else:
            n_e = get_electron_number_density(sw_data)[index]

        # if not self.shell_average:
        entropy = kb * sw_data.gas.temperatures[index] / (n_e ** (2 / 3))
        entropy.convert_to_units('keV*cm**2')

        entropy_profile = histogram_unyt(
            radial_distance,
            bins=lbins,
            weights=entropy,
            normalizer=weighting
        )
        entropy_profile.convert_to_units('keV*cm**2')

        kBT500 = (
                G * mean_molecular_weight * m500 * mp / r500 / 2
        ).to('keV')

        K500 = (
                kBT500 / (500 * fb * critical_density / (mean_atomic_weight_per_free_electron * mp)) ** (2 / 3)
        ).to('keV*cm**2')
        print(radial_bin_centres, entropy_profile, K500)
        return radial_bin_centres, entropy_profile, K500

    def plot_observations(self, axes: plt.Axes):
        sun_observations = Sun2009()
        r_r500, S_S500_50, S_S500_10, S_S500_90 = sun_observations.get_shortcut()

        rexcess = Pratt2010(n_radial_bins=21)
        bin_median, bin_perc16, bin_perc84 = rexcess.combine_entropy_profiles(
            m500_limits=(
                1e14 * Solar_Mass,
                5e14 * Solar_Mass
            ),
            k500_rescale=True
        )

        r = np.array([*axes.get_xlim()])
        k = 1.40 * r ** 1.1
        axes.plot(r, k, c='grey', ls='--', label='VKB (2005)')

        asymmetric_error = np.array(list(zip(
            bin_median - bin_perc16,
            bin_perc84 - bin_median
        ))).T
        axes.errorbar(
            rexcess.radial_bins,
            bin_median,
            yerr=asymmetric_error,
            fmt='o',
            markersize=2,
            color='grey',
            ecolor='lightgray',
            elinewidth=0.7,
            capsize=0,
            label=rexcess.citation
        )
        asymmetric_error = np.array(list(zip(
            S_S500_50 - S_S500_10,
            S_S500_90 - S_S500_50
        ))).T
        axes.errorbar(
            r_r500,
            S_S500_50,
            yerr=asymmetric_error,
            fmt='^',
            markersize=2,
            color='grey',
            ecolor='lightgray',
            elinewidth=0.7,
            capsize=0,
            label=sun_observations.citation
        )

    def process_catalogue(self):

        catalogue = self._process_catalogue(self.process_single_halo, labels=self.labels)
        # self.dump_to_pickle(self.filename, catalogue)
        return catalogue

    def read_catalogue(self):

        return self._read_catalogue(self.filename)

    def get_zoom_from_catalogue(self, **kwargs):

        return self._get_zoom_from_catalogue(self.filename, **kwargs)
