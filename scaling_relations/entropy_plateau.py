import os.path
import sys
import numpy as np
from swiftsimio import cosmo_array
from typing import Optional
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
from .electron_number_density import get_electron_number_density
from literature import Cosmology

sys.path.append("../xray")
import cloudy_softband as cloudy


class EntropyPlateau(HaloProperty):

    def __init__(
            self,
            max_radius_r500: float = 4,
            simple_electron_number_density: bool = False,
    ):
        super().__init__()

        self.max_radius_r500 = max_radius_r500
        self.simple_electron_number_density = simple_electron_number_density
        self.auto_masked = False

    def setup_data(
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
        self.fb = Cosmology().fb0
        self.critical_density = unyt_quantity(
            sw_data.metadata.cosmology.critical_density(sw_data.metadata.z).value, 'g/cm**3'
        ).to('Msun/Mpc**3')
        self.z = sw_data.metadata.z

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

        try:
            _ = sw_data.gas.heated_by_agnfeedback
        except AttributeError as err:
            print(err)
            if xlargs.debug:
                print(f"[{self.__class__.__name__}] Setting all agn_flag to zero.")
            sw_data.gas.heated_by_agnfeedback = np.zeros_like(sw_data.gas.densities)

        try:
            _ = sw_data.gas.heated_by_sniifeedback
        except AttributeError as err:
            print(err)
            if xlargs.debug:
                print(f"[{self.__class__.__name__}] Setting all snii_flag to zero.")
            sw_data.gas.heated_by_sniifeedback = np.zeros_like(sw_data.gas.densities)

        self.r500 = r500
        self.m500 = m500
        self.sw_data = sw_data

    def select_particles_on_plateau(
            self,
            shell_radius_r500: float = 0.1,
            shell_thickness_r500: float = 0.01,
            particle_ids: Optional[np.ndarray] = None,
            apply_mask: bool = True
    ):

        radial_distance = self.sw_data.gas.radial_distances / self.r500

        intersect_ids = np.ones_like(radial_distance, dtype=np.bool)
        if particle_ids is not None:
            _, match_indices, _ = np.intersect1d(
                self.sw_data.gas.particle_ids.value, particle_ids, assume_unique=True, return_indices=True
            )
            intersect_ids[~match_indices] = False

        print('intersect_ids', intersect_ids)
        shell_mask = np.where(
            (radial_distance > shell_radius_r500 - shell_thickness_r500 / 2) &
            (radial_distance < shell_radius_r500 + shell_thickness_r500 / 2) &
            (self.sw_data.gas.fofgroup_ids == 1) &
            (intersect_ids == True)
        )[0]

        print('shell_mask', shell_mask)

        if apply_mask:
            datasets_to_mask = []
            for dataset_name in dir(self.sw_data.gas):
                if not dataset_name.startswith('_') and not callable(getattr(self.sw_data.gas, dataset_name)):
                    datasets_to_mask.append(dataset_name)

            for dataset in datasets_to_mask:

                # Named columns treated in a separate loop
                if dataset == 'element_mass_fractions':
                    for element in self.sw_data.gas.element_mass_fractions.named_columns:
                        d = getattr(self.sw_data.gas.element_mass_fractions, element)
                        d = cosmo_array(
                            d[shell_mask].value,
                            units=d.units,
                            cosmo_factor=d.cosmo_factor,
                            comoving=d.comoving
                        )
                        setattr(self.sw_data.gas.element_mass_fractions, element, d)

                # Test whether the attribute contains particle data
                elif hasattr(getattr(self.sw_data.gas, dataset), 'value'):
                    d = getattr(self.sw_data.gas, dataset)
                    d = cosmo_array(
                        d[shell_mask].value,
                        units=d.units,
                        cosmo_factor=d.cosmo_factor,
                        comoving=d.comoving
                    )
                    setattr(self.sw_data.gas, dataset, d)

            del d, datasets_to_mask
            self.auto_masked = True

        else:
            self.shell_mask = shell_mask

        del shell_mask

    def shell_properties(self):

        if self.simple_electron_number_density:
            self.sw_data.gas.densities.convert_to_units('g/cm**3')
            n_e = self.sw_data.gas.densities / (mp * mean_atomic_weight_per_free_electron)
            n_e.convert_to_units('cm**-3')
        else:
            n_e = get_electron_number_density(self.sw_data)

        # if not self.shell_average:
        entropy = kb * self.sw_data.gas.temperatures / (n_e ** (2 / 3))
        entropy.convert_to_units('keV*cm**2')

        # Compute hydrogen number density and the log10
        # of the temperature to provide to the xray interpolator.
        data_nH = np.log10(
            self.sw_data.gas.element_mass_fractions.hydrogen * self.sw_data.gas.densities.to('g*cm**-3') / mp
        )
        data_T = np.log10(self.sw_data.gas.temperatures.value)

        # Interpolate the Cloudy table to get emissivities
        emissivities = unyt_array(
            10 ** cloudy.interpolate_X_Ray(
                data_nH,
                data_T,
                self.sw_data.gas.element_mass_fractions,
                fill_value=-50.
            ), 'erg/s/cm**3'
        )
        xray_luminosities = emissivities * self.sw_data.gas.masses / self.sw_data.gas.densities
        entropy_weighted_xray = np.average(entropy, weights=xray_luminosities)
        temperature_weighted_xray = np.average(self.sw_data.gas.temperatures, weights=xray_luminosities)
        density_weighted_xray = np.average(self.sw_data.gas.densities, weights=xray_luminosities)
        del data_nH, data_T, emissivities, xray_luminosities

        entropy_weighted_mass = np.average(entropy, weights=self.sw_data.gas.masses)
        temperature_weighted_mass = np.average(self.sw_data.gas.temperatures, weights=self.sw_data.gas.masses)
        density_weighted_mass = np.average(self.sw_data.gas.densities, weights=self.sw_data.gas.masses)

        volume_proxy = self.sw_data.gas.masses / self.sw_data.gas.densities
        entropy_weighted_volume = np.average(entropy, weights=volume_proxy)
        temperature_weighted_volume = np.average(self.sw_data.gas.temperatures, weights=volume_proxy)
        density_weighted_volume = np.average(self.sw_data.gas.densities, weights=volume_proxy)
        del volume_proxy

        kBT500 = (
                G * mean_molecular_weight * self.m500 * mp / self.r500 / 2
        ).to('keV')

        K500 = (
                kBT500 / (500 * self.fb * self.critical_density / (mean_atomic_weight_per_free_electron * mp)) ** (
                2 / 3)
        ).to('keV*cm**2')

        self.particle_entropies = entropy

        self.entropy_weighted_xray = entropy_weighted_xray
        self.entropy_weighted_mass = entropy_weighted_mass
        self.entropy_weighted_volume = entropy_weighted_volume

        self.temperature_weighted_xray = (temperature_weighted_xray * kb).to('keV')
        self.temperature_weighted_mass = (temperature_weighted_mass * kb).to('keV')
        self.temperature_weighted_volume = (temperature_weighted_volume * kb).to('keV')

        self.density_weighted_xray = density_weighted_xray
        self.density_weighted_mass = density_weighted_mass
        self.density_weighted_volume = density_weighted_volume

        self.kBT500 = kBT500
        self.K500 = K500

    def heating_fractions(self, nbins: int = 30):

        agn_flag = self.sw_data.gas.heated_by_agnfeedback > 0
        snii_flag = self.sw_data.gas.heated_by_sniifeedback > 0

        self.entropy_bin_edges = np.logspace(
            np.log10(self.particle_entropies.min().value),
            np.log10(self.particle_entropies.max().value),
            nbins + 1
        )
        self.entropy_hist, _ = np.histogram(
            self.particle_entropies,
            bins=self.entropy_bin_edges
        )
        self.entropy_hist_agn, _ = np.histogram(
            self.particle_entropies[agn_flag],
            bins=self.entropy_bin_edges
        )
        self.entropy_hist_snii, _ = np.histogram(
            self.particle_entropies[snii_flag],
            bins=self.entropy_bin_edges
        )
        self.entropy_hist_null, _ = np.histogram(
            self.particle_entropies[(~agn_flag & ~snii_flag)],
            bins=self.entropy_bin_edges
        )

        self.temperature_bin_edges = np.logspace(
            np.log10(self.sw_data.gas.temperatures.min().value),
            np.log10(self.sw_data.gas.temperatures.max().value),
            nbins + 1
        )
        self.temperature_hist, _ = np.histogram(
            self.sw_data.gas.temperatures,
            bins=self.temperature_bin_edges
        )
        self.temperature_hist_agn, _ = np.histogram(
            self.sw_data.gas.temperatures[agn_flag],
            bins=self.temperature_bin_edges
        )
        self.temperature_hist_snii, _ = np.histogram(
            self.sw_data.gas.temperatures[snii_flag],
            bins=self.temperature_bin_edges
        )
        self.temperature_hist_null, _ = np.histogram(
            self.sw_data.gas.temperatures[(~agn_flag & ~snii_flag)],
            bins=self.temperature_bin_edges
        )

        self.density_bin_edges = np.logspace(
            np.log10(self.sw_data.gas.densities.min().value),
            np.log10(self.sw_data.gas.densities.max().value),
            nbins + 1
        )
        self.density_hist, _ = np.histogram(
            self.sw_data.gas.densities,
            bins=self.density_bin_edges
        )
        self.density_hist_agn, _ = np.histogram(
            self.sw_data.gas.densities[agn_flag],
            bins=self.density_bin_edges
        )
        self.density_hist_snii, _ = np.histogram(
            self.sw_data.gas.densities[snii_flag],
            bins=self.density_bin_edges
        )
        self.density_hist_null, _ = np.histogram(
            self.sw_data.gas.densities[(~agn_flag & ~snii_flag)],
            bins=self.density_bin_edges
        )

        self.number_particles = len(self.particle_entropies)
        self.number_agn_heated = agn_flag.sum()
        self.number_snii_heated = snii_flag.sum()
        self.number_not_heated = self.number_particles - self.number_agn_heated - self.number_snii_heated

    def plot_observations(self, axes: plt.Axes):

        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.step(self.entropy_bin_edges[:-1], self.entropy_hist, label=f'All ({self.number_particles:d} particles)')
        axes.step(self.entropy_bin_edges[:-1], self.entropy_hist_null,
                  label=f'Not heated ({self.number_not_heated / self.number_particles * 100:.1f} %)')
        axes.step(self.entropy_bin_edges[:-1], self.entropy_hist_snii,
                  label=f'SN ({self.number_snii_heated / self.number_particles * 100:.1f} %)')
        axes.step(self.entropy_bin_edges[:-1], self.entropy_hist_agn,
                  label=f'AGN ({self.number_agn_heated / self.number_particles * 100:.1f} %)')
        axes.axvline(self.entropy_weighted_mass, linestyle=':')
        axes.axvline(self.K500, linestyle='--')
        axes.set_xlabel(r"$K$ [keV cm$^2$]")
        axes.set_ylabel(f"Number of particles")
        axes.legend(loc="upper right")
