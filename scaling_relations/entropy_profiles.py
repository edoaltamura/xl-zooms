import os.path
import sys
import numpy as np
from warnings import warn
from matplotlib import pyplot as plt
import scipy.stats as stat
import numba
from multiprocessing import cpu_count

numba.config.NUMBA_NUM_THREADS = cpu_count()

from unyt import (
    unyt_array,
    unyt_quantity,
    mh, G, mp, K, kb, cm, Solar_Mass, Mpc
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
from .electron_number_density import get_electron_number_density
from literature import Cosmology, Sun2009, Pratt2010

sys.path.append("../xray")
import cloudy_softband as cloudy


# def normalized_mean(r, quantity, normalizer, bins):
#     mean_value, _, _ = stat.binned_statistic(
#         x=r, values=quantity * normalizer, statistic="sum", bins=bins
#     )
#
#     normalization, _, _ = stat.binned_statistic(
#         x=r, values=normalizer, statistic="sum", bins=bins
#     )
#     if xlargs.debug:
#         print(mean_value, normalization)
#
#     return mean_value / normalization

def normalized_mean(r, quantity, normalizer, bins):
    mean_value, _ = histogram_unyt(
        r, bins=bins, weights=quantity * normalizer
    )

    normalization, _ = histogram_unyt(
        r, bins=bins, weights=normalizer
    )
    if xlargs.debug:
        print(mean_value, normalization)

    return mean_value / normalization


class EntropyProfiles(HaloProperty):

    def __init__(
            self,
            max_radius_r500: float = 4,
            xray_weighting: bool = True,
            simple_electron_number_density: bool = False,
    ):
        super().__init__()

        self.labels = ['radial_bin_centres', 'entropy_profile', 'K500']
        self.max_radius_r500 = max_radius_r500
        self.xray_weighting = xray_weighting
        self.simple_electron_number_density = simple_electron_number_density

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
        fb = Cosmology().get_baryon_fraction(sw_data.metadata.z)
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
            ).interpolate_hse()
            r500 = true_hse.r500hse
            m500 = true_hse.m500hse

        try:
            temperature = sw_data.gas.temperatures
        except AttributeError as err:
            print(f'[{self.__class__.__name__}] {err}')
            if xlargs.debug:
                print(f"[{self.__class__.__name__}] Computing gas temperature from internal energies.")
            A = sw_data.gas.entropies * sw_data.units.mass
            temperature = mean_molecular_weight * (gamma - 1) * (A * sw_data.gas.densities ** (5 / 3 - 1)) / (
                    gamma - 1) * mh / kb

        try:
            fof_ids = sw_data.gas.fofgroup_ids
        except AttributeError as err:
            print(f'[{self.__class__.__name__}] {err}')
            if xlargs.debug:
                print(f"[{self.__class__.__name__}] Select particles only by radial distance.")
            fof_ids = np.ones_like(sw_data.gas.densities)

        index = np.where(
            (sw_data.gas.radial_distances < self.max_radius_r500 * r500) &
            (fof_ids == 1) &
            (temperature > 1e5)
        )[0]
        radial_distance = sw_data.gas.radial_distances[index] / r500
        masses = sw_data.gas.masses[index]
        temperature = temperature[index]

        if self.simple_electron_number_density:
            n_e = sw_data.gas.densities.to('g/cm**3') / (mp * mean_molecular_weight)
        else:
            n_e = get_electron_number_density(sw_data)

        n_e = n_e.to('cm**-3')[index]

        # Define radial bins and shell volumes
        lbins = np.logspace(-2, np.log10(self.max_radius_r500), 51) * radial_distance.units
        radial_bin_centres = 10 ** (0.5 * np.log10(lbins[1:] * lbins[:-1])) * radial_distance.units

        if self.xray_weighting:

            entropy = kb * temperature / (n_e ** (2 / 3))
            entropy.convert_to_units('keV*cm**2')

            # Compute hydrogen number density and the log10
            # of the temperature to provide to the xray interpolator.
            data_nH = np.log10(
                sw_data.gas.element_mass_fractions.hydrogen * sw_data.gas.densities.to('g*cm**-3') / mp)
            data_T = np.log10(sw_data.gas.temperatures.value)

            # Interpolate the Cloudy table to get emissivities
            emissivities = cloudy.interpolate_X_Ray(
                data_nH,
                data_T,
                sw_data.gas.element_mass_fractions
            )
            log10_Mpc3_to_cm3 = np.log10(Mpc.get_conversion_factor(cm)[0] ** 3)
            emissivities = unyt_quantity(
                10 ** (
                        cloudy.logsumexp(
                            emissivities[index],
                            b=(sw_data.gas.masses[index] / sw_data.gas.densities[index]).value,
                            base=10.
                        ) + log10_Mpc3_to_cm3
                ), 'erg/s'
            )

            if xlargs.debug:
                print(f'emissivities.shape: {emissivities.shape}\n{emissivities}')

            entropy_profile = normalized_mean(
                radial_distance, entropy, emissivities, lbins
            )

        else:
            volume_shell = (4. * np.pi / 3.) * (r500 ** 3) * ((lbins[1:]) ** 3 - (lbins[:-1]) ** 3)

            mass_weights, _ = histogram_unyt(radial_distance, bins=lbins, weights=masses)
            mass_weights[mass_weights == 0] = np.nan  # Replace zeros with Nans
            density_profile = mass_weights / volume_shell
            number_density_profile = (density_profile.to('g/cm**3') / (mp * mean_molecular_weight)).to('cm**-3')

            mass_weighted_temperatures = (temperature * kb).to('keV') * masses
            temperature_weights, _ = histogram_unyt(radial_distance, bins=lbins, weights=mass_weighted_temperatures)
            temperature_weights[temperature_weights == 0] = np.nan  # Replace zeros with Nans
            temperature_profile = temperature_weights / mass_weights  # kBT in units of [keV]

            entropy_profile = temperature_profile / number_density_profile ** (2 / 3)
            density_profile /= critical_density

        kBT500 = (
                G * mean_molecular_weight * m500 * mp / r500 / 2
        ).to('keV')

        K500 = (
                kBT500 / (500 * fb * critical_density / (mean_atomic_weight_per_free_electron * mp)) ** (2 / 3)
        ).to('keV*cm**2')

        return radial_bin_centres, entropy_profile, K500

    def display_single_halo(self, *args, **kwargs):
        radial_bin_centres, entropy_profile, K500 = self.process_single_halo(*args, **kwargs)

        set_mnras_stylesheet()
        fig = plt.figure(constrained_layout=True)
        axes = fig.add_subplot()

        axes.plot(
            radial_bin_centres,
            entropy_profile / K500,
            linestyle='-',
            color='r',
            linewidth=1,
            alpha=1,
        )
        axes.set_xscale('log')
        axes.set_yscale('log')

        axes.axvline(0.15, color='k', linestyle='--', lw=0.5, zorder=0)
        axes.set_ylabel(r'Entropy [keV cm$^2$]')
        axes.set_xlabel(r'$r/r_{500}$')
        # axes[1, 2].set_ylim([1, 1e4])
        axes.set_ylim([1e-2, 5])
        axes.set_xlim([0.01, self.max_radius_r500])

        # axes[1, 2].axhline(y=K500, color='k', linestyle=':', linewidth=0.5)
        # axes[1, 2].text(
        #     axes[1, 2].get_xlim()[0], K500, r'$K_{500}$',
        #     horizontalalignment='left',
        #     verticalalignment='bottom',
        #     color='k',
        #     bbox=dict(
        #         boxstyle='square,pad=10',
        #         fc='none',
        #         ec='none'
        #     )
        # )

        sun_observations = Sun2009()
        sun_observations.filter_by('M_500', 8e13, 3e14)
        sun_observations.overlay_entropy_profiles(
            axes=axes,
            k_units='K500adi',
            markersize=1,
            linewidth=0.5
        )
        rexcess = Pratt2010()
        bin_median, bin_perc16, bin_perc84 = rexcess.combine_entropy_profiles(
            m500_limits=(
                1e14 * Solar_Mass,
                5e14 * Solar_Mass
            ),
            k500_rescale=True
        )
        axes.fill_between(
            rexcess.radial_bins,
            bin_perc16,
            bin_perc84,
            color='aqua',
            alpha=0.85,
            linewidth=0
        )
        axes.plot(rexcess.radial_bins, bin_median, c='k')

        if not xlargs.quiet:
            plt.show()

        plt.close()

    def process_catalogue(self):

        catalogue = self._process_catalogue(self.process_single_halo, labels=self.labels)
        self.dump_to_pickle(self.filename, catalogue)

    def read_catalogue(self):

        return self._read_catalogue(self.filename)

    def get_zoom_from_catalogue(self, **kwargs):

        return self._get_zoom_from_catalogue(self.filename, **kwargs)
