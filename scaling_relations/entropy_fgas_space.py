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
from .entropy_profiles import EntropyProfiles
from hydrostatic_estimates import HydrostaticEstimator
from literature import Cosmology, Sun2009, Pratt2010


def normalized_mean(r, quantity, normalizer, bins):
    mean_value = histogram_unyt(
        r, bins=bins, weights=quantity * normalizer
    )

    normalization = histogram_unyt(
        r, bins=bins, weights=normalizer
    )
    if xlargs.debug:
        print(mean_value, normalization)

    return mean_value / normalization


class EntropyFgasSpace(HaloProperty):

    def __init__(
            self,
            max_radius_r500: float = 4,
            xray_weighting: bool = True,
            simple_electron_number_density: bool = False,
            shell_average: bool = True
    ):
        super().__init__()

        self.labels = ['radial_bin_centres', 'entropy_profile', 'K500']
        self.max_radius_r500 = max_radius_r500
        self.xray_weighting = xray_weighting
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

    def get_simple_ne(self):
        pass

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
        setattr(self, 'fb', fb)

        try:
            r500 = vr_data.spherical_overdensities.r_500_rhocrit[0].to('Mpc')
        except AttributeError as err:
            print(f'[{self.__class__.__name__}] {err}')

            spherical_overdensity = SODelta500(
                path_to_snap=path_to_snap,
                path_to_catalogue=path_to_catalogue,
            )
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

        # Define radial bins
        lbins = np.logspace(-2, np.log10(self.max_radius_r500), 51) * dimensionless
        radial_bin_centres = 10 ** (0.5 * np.log10(lbins[1:] * lbins[:-1])) * dimensionless

        # Compute gas mass profile
        sw_data.gas.radial_distances.convert_to_physical()
        sw_data.gas.masses.convert_to_physical()
        masses = sw_data.gas.masses

        try:
            temperatures = sw_data.gas.temperatures
        except AttributeError as err:
            print(f'[{self.__class__.__name__}] {err}')
            if xlargs.debug:
                print(f"[{self.__class__.__name__}] Computing gas temperature from internal energies.")
            sw_data.gas.densities.convert_to_physical()
            A = sw_data.gas.entropies * sw_data.units.mass
            temperatures = mean_molecular_weight * (gamma - 1) * (A * sw_data.gas.densities ** (5 / 3 - 1)) / (
                    gamma - 1) * mh / kb

        try:
            fof_ids = sw_data.gas.fofgroup_ids

            # Select all particles within sphere
            mask = np.where(
                (sw_data.gas.radial_distances <= self.max_radius_r500 * r500) &
                (fof_ids == 1) &
                (temperatures > Tcut_halogas)
            )[0]

            del fof_ids

        except AttributeError as err:
            print(err)
            print(f"[{self.__class__.__name__}] Select particles only by radial distance.")
            mask = np.where(
                (sw_data.gas.radial_distances <= self.max_radius_r500 * r500) &
                (temperatures > Tcut_halogas)
            )[0]

        radial_distances = sw_data.gas.radial_distances[mask] / r500
        assert (radial_distances >= 0).all()
        masses = unyt_array(masses, sw_data.units.mass)[mask]
        assert (masses >= 0).all()
        del mask

        mass_weights = histogram_unyt(radial_distances, bins=lbins, weights=masses)
        mass_weights[mass_weights == 0] = np.nan  # Replace zeros with Nans
        cumulative_gas_mass_profile = np.nancumsum(mass_weights.value) * masses.units

        # Compute total mass profile
        sw_data.gas.radial_distances.convert_to_physical()
        sw_data.dark_matter.radial_distances.convert_to_physical()

        radial_distances_collect = [
            sw_data.gas.radial_distances,
            sw_data.dark_matter.radial_distances,
        ]
        if sw_data.metadata.n_stars > 0:
            sw_data.stars.radial_distances.convert_to_physical()
            radial_distances_collect.append(sw_data.stars.radial_distances)
        elif xlargs.debug:
            print(f"[{self.__class__.__name__}] stars not detected.")

        if sw_data.metadata.n_black_holes > 0:
            sw_data.black_holes.radial_distances.convert_to_physical()
            radial_distances_collect.append(sw_data.black_holes.radial_distances)
        elif xlargs.debug:
            print(f"[{self.__class__.__name__}] black_holes not detected.")

        radial_distances = np.concatenate(radial_distances_collect) * sw_data.units.length / r500

        sw_data.gas.masses.convert_to_physical()
        sw_data.dark_matter.masses.convert_to_physical()
        masses_collect = [
            sw_data.gas.masses,
            sw_data.dark_matter.masses,
        ]
        if sw_data.metadata.n_stars > 0:
            sw_data.stars.masses.convert_to_physical()
            masses_collect.append(sw_data.stars.masses)
        elif xlargs.debug:
            print(f"[{self.__class__.__name__}] stars not detected.")

        if sw_data.metadata.n_black_holes > 0:
            sw_data.black_holes.subgrid_masses.convert_to_physical()
            masses_collect.append(sw_data.black_holes.subgrid_masses)
        elif xlargs.debug:
            print(f"[{self.__class__.__name__}] black_holes not detected.")

        masses = np.concatenate(masses_collect)

        try:
            fof_ids_collect = [
                sw_data.gas.fofgroup_ids,
                sw_data.dark_matter.fofgroup_ids,
            ]
            if sw_data.metadata.n_stars > 0:
                fof_ids_collect.append(sw_data.stars.fofgroup_ids)
            elif xlargs.debug:
                print(f"[{self.__class__.__name__}] stars not detected.")

            if sw_data.metadata.n_black_holes > 0:
                fof_ids_collect.append(sw_data.black_holes.fofgroup_ids)
            elif xlargs.debug:
                print(f"[{self.__class__.__name__}] black_holes not detected.")

            fof_ids = np.concatenate(fof_ids_collect)

            # Select all particles within sphere
            mask = np.where(
                (radial_distances <= self.max_radius_r500) &
                (fof_ids == 1)
            )[0]

            del fof_ids

        except AttributeError as err:
            print(err)
            print(f"[{self.__class__.__name__}] Select particles only by radial distance.")
            mask = np.where(radial_distances <= self.max_radius_r500)[0]

        mass_weights = histogram_unyt(radial_distances[mask], bins=lbins, weights=masses[mask])
        mass_weights[mass_weights == 0] = np.nan  # Replace zeros with Nans
        cumulative_mass_profile = np.nancumsum(mass_weights.value) * sw_data.units.mass

        return radial_bin_centres, cumulative_gas_mass_profile, cumulative_mass_profile

    def display_single_halo(self, *args, **kwargs):

        # Get entropy profile
        entropy_profile_obj = EntropyProfiles(
            max_radius_r500=self.max_radius_r500,
            simple_electron_number_density=True,
            xray_weighting=False,
            shell_average=True
        )
        _, entropy_profile, K500 = entropy_profile_obj.process_single_halo(*args, **kwargs)
        entropy_profile /= K500
        print(entropy_profile)

        _, cumulative_gas_mass_profile, cumulative_mass_profile = self.process_single_halo(*args, **kwargs)
        gas_fraction_enclosed = cumulative_gas_mass_profile / cumulative_mass_profile
        print(gas_fraction_enclosed)

        set_mnras_stylesheet()
        fig = plt.figure(constrained_layout=True)
        axes = fig.add_subplot()

        axes.plot(
            gas_fraction_enclosed,
            entropy_profile,
            linestyle='-',
            color='r',
            linewidth=1,
            alpha=1,
        )
        axes.set_xscale('log')
        axes.set_yscale('log')

        axes.set_ylabel(r'Entropy [$K_{500}$]')
        axes.set_xlabel(r'$f_{\rm gas} = M_{\rm gas}(<r) / M_{\rm tot}(<r)$')
        axes.set_ylim([1e-2, 5])
        axes.set_xlim([1e-5, 1.1 * self.fb])

        axes.axvline(x=self.fb, color='k', linestyle=':', linewidth=0.5)
        axes.text(
            self.fb, axes.get_ylim()[1], r'Baryon fraction',
            horizontalalignment='left',
            verticalalignment='bottom',
            color='k',
            bbox=dict(
                boxstyle='square,pad=10',
                fc='none',
                ec='none'
            )
        )

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
