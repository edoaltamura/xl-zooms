import os.path
import numpy as np
from warnings import warn
from unyt import (
    unyt_array,
    unyt_quantity,
    mh, G, mp, K, kb, cm, Solar_Mass
)

from .halo_property import HaloProperty, histogram_unyt
from .spherical_overdensities import SODelta500
from hydrostatic_estimates import HydrostaticEstimator
from register import Zoom, Tcut_halogas, default_output_directory, args
from literature import Cosmology, Sun2009, Pratt2010


mean_molecular_weight = 0.59
mean_atomic_weight_per_free_electron = 1.14
primordial_hydrogen_mass_fraction = 0.76
solar_metallicity = 0.0133714
gamma = 5 / 3


class EntropyProfiles(HaloProperty):

    def __init__(self, max_radius_r500: float = 4):
        super().__init__()

        self.labels = ['radial_bin_centres', 'entropy_profile', 'K500']
        self.max_radius_r500 = max_radius_r500

        self.filename = os.path.join(
            default_output_directory,
            'intermediate',
            f'gas_fractions_{args.mass_estimator:s}_{args.redshift_index:04d}.pkl'
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

        if args.mass_estimator == 'hse':
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
            if args.debug:
                print(f"[{self.__class__.__name__}] Computing gas temperature from internal energies.")
            A = sw_data.gas.entropies * sw_data.units.mass
            temperature = mean_molecular_weight * (gamma - 1) * (A * sw_data.gas.densities ** (5 / 3 - 1)) / (
                    gamma - 1) * mh / kb

        try:
            fof_ids = sw_data.gas.fofgroup_ids
        except AttributeError as err:
            print(f'[{self.__class__.__name__}] {err}')
            if args.debug:
                print(f"[{self.__class__.__name__}] Select particles only by radial distance.")
            fof_ids = np.ones_like(sw_data.gas.densities)

        index = np.where(
            (sw_data.gas.radial_distances < self.max_radius_r500 * r500) &
            (fof_ids == 1) &
            (temperature > 1e5)
        )[0]
        radial_distance = sw_data.gas.radial_distances[index] / r500
        sw_data.gas.masses = sw_data.gas.masses[index]
        temperature = temperature[index]

        # Define radial bins and shell volumes
        lbins = np.logspace(-2, np.log10(self.max_radius_r500), 51) * radial_distance.units
        radial_bin_centres = 10 ** (0.5 * np.log10(lbins[1:] * lbins[:-1])) * radial_distance.units
        volume_shell = (4. * np.pi / 3.) * (r500 ** 3) * ((lbins[1:]) ** 3 - (lbins[:-1]) ** 3)

        mass_weights, _ = histogram_unyt(radial_distance, bins=lbins, weights=sw_data.gas.masses)
        mass_weights[mass_weights == 0] = np.nan  # Replace zeros with Nans
        density_profile = mass_weights / volume_shell
        number_density_profile = (density_profile.to('g/cm**3') / (mp * mean_molecular_weight)).to('cm**-3')

        mass_weighted_temperatures = (temperature * kb).to('keV') * sw_data.gas.masses
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

    def process_catalogue(self):

        catalogue = self._process_catalogue(self.process_single_halo, labels=self.labels)
        self.dump_to_pickle(self.filename, catalogue)

    def read_catalogue(self):

        return self._read_catalogue(self.filename)

    def get_zoom_from_catalogue(self, **kwargs):

        return self._get_zoom_from_catalogue(self.filename, **kwargs)
