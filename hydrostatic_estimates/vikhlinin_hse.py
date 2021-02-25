import sys
import os
import unyt
import numpy as np
from typing import Tuple
import h5py as h5
import swiftsimio as sw
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from scipy.interpolate import interp1d

# Make the register backend visible to the script
sys.path.append("../zooms")
sys.path.append("../observational_data")

from register import zooms_register, Zoom, Tcut_halogas, name_list
from convergence_radius import convergence_radius
import observational_data as obs


try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

cosmology = obs.Observations().cosmo_model
fbary = cosmology.Ob0 / cosmology.Om0  # Cosmic baryon fraction
mean_molecular_weight = 0.5954
true_data_nbins = 51


def histogram_unyt(
        data: unyt.unyt_array,
        bins: unyt.unyt_array = None,
        weights: unyt.unyt_array = None,
        **kwargs,
) -> Tuple[unyt.unyt_array]:
    assert data.shape == weights.shape, (
        "Data and weights arrays must have the same shape. "
        f"Detected data {data.shape}, weights {weights.shape}."
    )

    assert data.units == bins.units, (
        "Data and bins must have the same units. "
        f"Detected data {data.units}, bins {bins.units}."
    )

    hist, bin_edges = np.histogram(data.value, bins=bins.value, weights=weights.value, **kwargs)
    hist *= weights.units
    bin_edges *= data.units

    return hist, bin_edges


def cumsum_unyt(data: unyt.unyt_array, **kwargs) -> unyt.unyt_array:
    res = np.cumsum(data.value, **kwargs)

    return res * data.units


class HydrostaticDiagnostic:
    # This class is to be used for debugging purposes

    def __init__(self, zoom: Zoom):
        self.zoom = zoom
        self.total_mass_profiles()

    def total_mass_profiles(self):

        # Read in halo properties from catalog
        with h5.File(self.zoom.catalog_file, 'r') as h5file:
            self.R2500c = unyt.unyt_quantity(h5file['/SO_R_2500_rhocrit'][0], unyt.Mpc)
            self.R500c = unyt.unyt_quantity(h5file['/SO_R_500_rhocrit'][0], unyt.Mpc)
            XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc)
            YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc)
            ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc)

        # Read in gas particles and parse densities and temperatures
        mask = sw.mask(self.zoom.snapshot_file, spatial_only=False)
        region = [[XPotMin - 1.5 * self.R500c, XPotMin + 1.5 * self.R500c],
                  [YPotMin - 1.5 * self.R500c, YPotMin + 1.5 * self.R500c],
                  [ZPotMin - 1.5 * self.R500c, ZPotMin + 1.5 * self.R500c]]
        mask.constrain_spatial(region)
        mask.constrain_mask(
            "gas", "temperatures",
            Tcut_halogas * mask.units.temperature,
            1.e12 * mask.units.temperature
        )
        data = sw.load(self.zoom.snapshot_file, mask=mask)

        # Set bounds for the radial profiles
        radius_bounds = [0.15, 1.5]
        lbins = np.logspace(
            np.log10(radius_bounds[0]), np.log10(radius_bounds[1]), true_data_nbins
        ) * unyt.dimensionless

        shell_volume = (4 / 3 * np.pi) * self.R500c ** 3 * (lbins[1:] ** 3 - lbins[:-1] ** 3)

        unitLength = data.metadata.units.length
        unitMass = data.metadata.units.mass
        critical_density = unyt.unyt_quantity(
            data.metadata.cosmology_raw['Critical density [internal units]'],
            unitMass / unitLength ** 3
        )[0].to('Msun/Mpc**3')

        # Select hot gas within sphere and without core
        deltaX = data.gas.coordinates[:, 0] - XPotMin
        deltaY = data.gas.coordinates[:, 1] - YPotMin
        deltaZ = data.gas.coordinates[:, 2] - ZPotMin
        deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) / self.R500c

        # Keep only particles inside 1.5 R500crit
        index = np.where(deltaR < radius_bounds[1])[0]
        central_mass = sum(data.gas.masses[np.where(deltaR < radius_bounds[0])[0]])
        mass_weights, _ = histogram_unyt(deltaR[index], bins=lbins, weights=data.gas.masses[index])

        self.density_profile_input = mass_weights / shell_volume / critical_density

        # Select DM within sphere and without core
        deltaX = data.dark_matter.coordinates[:, 0] - XPotMin
        deltaY = data.dark_matter.coordinates[:, 1] - YPotMin
        deltaZ = data.dark_matter.coordinates[:, 2] - ZPotMin
        deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) / self.R500c

        # Keep only particles inside 1.5 R500crit
        index = np.where(deltaR < radius_bounds[1])[0]
        central_mass += sum(data.dark_matter.masses[np.where(deltaR < radius_bounds[0])[0]])
        _mass_weights, _ = histogram_unyt(deltaR[index], bins=lbins, weights=data.dark_matter.masses[index])
        mass_weights += _mass_weights

        # Select stars within sphere and without core
        deltaX = data.stars.coordinates[:, 0] - XPotMin
        deltaY = data.stars.coordinates[:, 1] - YPotMin
        deltaZ = data.stars.coordinates[:, 2] - ZPotMin
        deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) / self.R500c

        # Keep only particles inside 1.5 R500crit
        index = np.where(deltaR < radius_bounds[1])[0]
        central_mass += sum(data.stars.masses[np.where(deltaR < radius_bounds[0])[0]])
        _mass_weights, _ = histogram_unyt(deltaR[index], bins=lbins, weights=data.stars.masses[index])
        mass_weights += _mass_weights

        # Replace zeros with Nans
        mass_weights[mass_weights == 0] = np.nan
        cumulative_mass = central_mass + cumsum_unyt(mass_weights)

        self.radial_bin_centres_input = 10.0 ** (0.5 * np.log10(lbins[1:] * lbins[:-1])) * unyt.dimensionless
        self.cumulative_mass_input = cumulative_mass.to('Msun')
        self.total_density_profile_input = mass_weights / shell_volume / critical_density

    def plot_all(self):
        fields = [
            'density_profile',
            # 'temperature_profile',
            # 'cumulative_mass',
            # 'total_density_profile',
        ]
        labels = [
            r'$\rho_{\rm gas} / \rho_{\rm crit}$',
            # r'$k_{\rm B}T$ [keV]',
            # r'$M(<R)$ [M$_\odot$]',
            # r'$\rho / \rho_{\rm crit}$',
        ]
        for i, (field, label) in enumerate(zip(fields, labels)):
            print(f"({i + 1}/{len(fields)}) Generating diagnostic plot: {field}.")
            self.plot_profile(
                field_name=field,
                ylabel=label,
                filename=f"diagnostic_{field}_{self.zoom.run_name}.png"
            )

    def plot_profile(self, field_name: str = 'radial_bin_centres',
                     ylabel: str = 'y', filename: str = 'diagnostics.png'):

        fig, (ax, ax_residual) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(3, 4),
            sharex=True,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.02}
        )

        x_input = self.radial_bin_centres_input
        x_hse = self.radial_bin_centres_hse
        y_input = getattr(self, field_name + '_input')
        y_hse = getattr(self, field_name + '_hse')

        ax.plot(x_input, y_input, label="True data", color='lime')
        ax.plot(x_hse, y_hse, label=f"Vikhlinin HSE fit from {self.profile_type} data", color='orange')
        ax_residual.plot(x_hse, 1 - y_hse / y_input, color='orange')
        ax_residual.plot([x_hse.min(), x_hse.max()], [0, 0], color='lime')

        # If plotting cumulative mass, display HSE biases
        if field_name == 'cumulative_mass':
            ax.axvline(x=1, color='lime', linestyle='--')
            ax.axvline(x=self.R500hse / self.R500c, color='orange', linestyle='--')
            y_center = 10.0 ** (0.5 * np.log10(y_input.max() * y_input.max()))
            ax.text(1.05, y_center, r"$R_{500, \rm crit}$", rotation=90, va='center', ha='left', color='grey')
            ax.text(self.R500hse / self.R500c * 0.95, y_center, r"$R_{500, \rm hse}$", rotation=90,
                    va='center', ha='right', color='grey')

            ax_residual.plot([self.R2500c / self.R500c] * 2, [0, self.b2500hse], color='grey', marker='.')
            ax_residual.plot([1, 1], [0, self.b500hse], color='grey', marker='.')
            ax_residual.text(self.R2500c / self.R500c * 1.05, self.b2500hse / 2,
                             f"$b_{{\\rm 2500,hse}}$\n{self.b2500hse.v:.3f}", color='grey')
            ax_residual.text(1.05, self.b500hse / 2,
                             f"$b_{{\\rm 500,hse}}$\n{self.b500hse.v:.3f}", color='grey')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(ylabel, fontsize='large')
        ax_residual.set_ylabel(r"$1-({\rm hse} / {\rm true})$", fontsize='large')
        ax_residual.set_xlabel(r"$R\ /\ R_{\rm 500c\ (true)}$", fontsize='large')
        ax.legend()
        ax.set_title(f"{self.zoom.run_name}", fontsize=5)
        plt.tight_layout()
        # plt.savefig(f"{self.output_directory}/hse_diagnostic/{filename}", bbox_inches="tight")
        plt.show()
        plt.close()


class HydrostaticEstimator:

    def __init__(self, zoom: Zoom, excise_core: bool = True, profile_type: str = 'true',
                 using_mcmc: bool = False, spec_fit_data: dict = None, diagnostics_on: bool = False):
        self.zoom = zoom
        self.using_mcmc = using_mcmc
        self.excise_core = excise_core
        self.profile_type = profile_type
        self.diagnostics_on = diagnostics_on

        # Initialise an HydrostaticDiagnostic instance
        # Parse to this objects all profiles and quantities for external checks
        if self.diagnostics_on:
            self.diagnostics = HydrostaticDiagnostic(zoom)

        if profile_type.lower() == 'true':
            self.load_zoom_profiles()
        elif profile_type.lower() == 'spec' or 'xray' in profile_type.lower():
            assert spec_fit_data, "Spec output data not detected."
            self.load_xray_profiles(spec_fit_data)

        # Parse fitted profiles to diagnostic container
        if self.diagnostics_on:
            setattr(self.diagnostics, 'profile_type', self.profile_type)
            setattr(self.diagnostics, 'output_directory', self.zoom.output_directory)
            setattr(self.diagnostics, 'temperature_profile_input', self.temperature_profile)

        self.interpolate_hse()

    @classmethod
    def from_data_paths(cls, catalog_file: str, snapshot_file: str,
                        excise_core: bool = True, profile_type: str = 'true',
                        using_mcmc: bool = False, spec_fit_data: dict = None,
                        diagnostics_on: bool = False):
        """
        If you wish not to parse a Zoom object, but the absolute
        paths for the snapshot and VR catalogue, you may use this
        class method, which reconstructs the Zoom object from the
        known zooms register and returns an HydrostaticEstimator
        instance as normal.
        """
        zoom_found = None
        for zoom in zooms_register:
            if zoom.catalog_file == catalog_file and zoom.snapshot_file == snapshot_file:
                zoom_found = zoom
                break

        assert zoom_found, (
            f"The catalogue ({catalog_file}) and snapshot ({snapshot_file}) "
            "paths cannot be found in registered zooms."
        )

        return cls(zoom_found, excise_core=excise_core, profile_type=profile_type,
                   using_mcmc=using_mcmc, spec_fit_data=spec_fit_data, diagnostics_on=diagnostics_on)

    def load_zoom_profiles(self):
        # Read in halo properties from catalog
        with h5.File(self.zoom.catalog_file, 'r') as h5file:
            self.M2500c = unyt.unyt_quantity(h5file['/SO_Mass_2500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
            self.M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
            self.M200c = unyt.unyt_quantity(h5file['/Mass_200crit'][0] * 1.e10, unyt.Solar_Mass)
            self.R500c = unyt.unyt_quantity(h5file['/SO_R_500_rhocrit'][0], unyt.Mpc)
            XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc)
            YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc)
            ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc)

        # Read in gas particles and parse densities and temperatures
        mask = sw.mask(self.zoom.snapshot_file, spatial_only=False)
        region = [[XPotMin - 5 * self.R500c, XPotMin + 5 * self.R500c],
                  [YPotMin - 5 * self.R500c, YPotMin + 5 * self.R500c],
                  [ZPotMin - 5 * self.R500c, ZPotMin + 5 * self.R500c]]
        mask.constrain_spatial(region)
        mask.constrain_mask(
            "gas", "temperatures",
            Tcut_halogas * mask.units.temperature,
            1.e12 * mask.units.temperature
        )
        data = sw.load(self.zoom.snapshot_file, mask=mask)

        # Calculate the critical density for the density profile
        unitLength = data.metadata.units.length
        unitMass = data.metadata.units.mass
        self.rho_crit = unyt.unyt_quantity(
            data.metadata.cosmology_raw['Critical density [internal units]'],
            unitMass / unitLength ** 3
        )[0].to('Msun/Mpc**3')

        # Select hot gas within sphere and without core
        deltaX = data.gas.coordinates[:, 0] - XPotMin
        deltaY = data.gas.coordinates[:, 1] - YPotMin
        deltaZ = data.gas.coordinates[:, 2] - ZPotMin
        deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2)

        # Set bounds for the radial profiles
        radius_bounds = [0.15, 1.5]

        # Keep only particles inside 5 R500crit
        index = np.where(deltaR < radius_bounds[1] * self.R500c)[0]
        radial_distance_scaled = deltaR[index] / self.R500c
        assert radial_distance_scaled.units == unyt.dimensionless
        gas_masses = data.gas.masses[index]
        gas_temperatures = data.gas.temperatures[index]
        gas_mass_weighted_temperatures = gas_temperatures * gas_masses

        if not self.excise_core:
            # Compute convergence radius and set as inner limit
            gas_convergence_radius = convergence_radius(
                deltaR[index],
                gas_masses.to('Msun'),
                self.rho_crit.to('Msun/Mpc**3')
            ) / self.R500c
            radius_bounds[0] = gas_convergence_radius.value

        lbins = np.logspace(
            np.log10(radius_bounds[0]), np.log10(radius_bounds[1]), true_data_nbins
        ) * radial_distance_scaled.units

        mass_weights, bin_edges = histogram_unyt(radial_distance_scaled, bins=lbins, weights=gas_masses)

        # Replace zeros with Nans
        mass_weights[mass_weights == 0] = np.nan

        # Set the radial bins as object attribute
        self.radial_bin_centres = 10.0 ** (0.5 * np.log10(lbins[1:] * lbins[:-1])) * unyt.dimensionless
        self.radial_bin_edges = lbins

        # Compute the radial gas density profile
        volume_shell = (4. * np.pi / 3.) * (self.R500c ** 3) * ((bin_edges[1:]) ** 3 - (bin_edges[:-1]) ** 3)
        self.density_profile = mass_weights / volume_shell / self.rho_crit
        assert self.density_profile.units == unyt.dimensionless

        # Compute the radial mass-weighted temperature profile
        hist, _ = histogram_unyt(radial_distance_scaled, bins=lbins, weights=gas_mass_weighted_temperatures)
        hist /= mass_weights
        self.temperature_profile = (hist * unyt.boltzmann_constant).to('keV')

    def load_xray_profiles(self, spec_fit_data: dict):

        # Read in halo properties from catalog
        with h5.File(self.zoom.catalog_file, 'r') as h5file:
            self.M2500c = unyt.unyt_quantity(h5file['/SO_Mass_2500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
            self.M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
            self.M200c = unyt.unyt_quantity(h5file['/Mass_200crit'][0] * 1.e10, unyt.Solar_Mass)
            self.R500c = unyt.unyt_quantity(h5file['/SO_R_500_rhocrit'][0], unyt.Mpc)
            XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc)
            YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc)
            ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc)

        # Read in gas particles and parse densities and temperatures
        mask = sw.mask(self.zoom.snapshot_file, spatial_only=False)
        region = [[XPotMin - 5 * self.R500c, XPotMin + 5 * self.R500c],
                  [YPotMin - 5 * self.R500c, YPotMin + 5 * self.R500c],
                  [ZPotMin - 5 * self.R500c, ZPotMin + 5 * self.R500c]]
        mask.constrain_spatial(region)
        mask.constrain_mask(
            "gas", "temperatures",
            Tcut_halogas * mask.units.temperature,
            1.e12 * mask.units.temperature
        )
        data = sw.load(self.zoom.snapshot_file, mask=mask)

        # Calculate the critical density for the density profile
        unitLength = data.metadata.units.length
        unitMass = data.metadata.units.mass
        self.rho_crit = unyt.unyt_quantity(
            data.metadata.cosmology_raw['Critical density [internal units]'],
            unitMass / unitLength ** 3
        )[0].to('Msun/Mpc**3')

        # Select hot gas within sphere and without core
        deltaX = data.gas.coordinates[:, 0] - XPotMin
        deltaY = data.gas.coordinates[:, 1] - YPotMin
        deltaZ = data.gas.coordinates[:, 2] - ZPotMin
        deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2)

        # Keep only particles inside 5 R500crit
        index = np.where(deltaR < 5 * self.R500c)[0]
        radial_distance_scaled = deltaR[index] / self.R500c
        assert radial_distance_scaled.units == unyt.dimensionless
        gas_masses = data.gas.masses[index]

        # Set bounds for the radial profiles
        radius_bounds = [0.15, 5]
        if not self.excise_core:
            # Compute convergence radius and set as inner limit
            gas_convergence_radius = convergence_radius(
                deltaR[index],
                gas_masses.to('Msun'),
                self.rho_crit.to('Msun/Mpc**3')
            ) / self.R500c
            radius_bounds[0] = gas_convergence_radius.value

        # Create the interpolation objects for the x-ray density and temperature
        assert 'Rspec' in spec_fit_data, f"`Rspec` key not found in the spec data fitting output: {spec_fit_data}."
        assert 'RHOspec' in spec_fit_data, f"`RHOspec` key not found in the spec data fitting output: {spec_fit_data}."
        assert 'Tspec' in spec_fit_data, f"`Tspec` key not found in the spec data fitting output: {spec_fit_data}."

        spec_density_interpolate = interp1d(spec_fit_data['Rspec'], spec_fit_data['RHOspec'], kind='linear')
        spec_temperature_interpolate = interp1d(spec_fit_data['Rspec'], spec_fit_data['Tspec'], kind='linear')

        lbins = np.logspace(
            np.log10(radius_bounds[0]), np.log10(radius_bounds[1]), true_data_nbins
        ) * radial_distance_scaled.units

        self.radial_bin_centres = 10.0 ** (0.5 * np.log10(lbins[1:] * lbins[:-1])) * unyt.dimensionless
        self.radial_bin_edges = lbins

        # Cut ends of the radial bins to interpolate, since they might be
        # outside the spec_fit_data['Rspec'] range
        # Prevents ValueError: A value in x_new is below the interpolation range.
        radial_bins_intersect = np.where(
            (self.radial_bin_centres * self.R500c > spec_fit_data['Rspec'].min()) &
            (self.radial_bin_centres * self.R500c < spec_fit_data['Rspec'].max())
        )[0]
        self.radial_bin_centres = self.radial_bin_centres[radial_bins_intersect]

        # Compute the radial gas density profile
        self.density_profile = spec_density_interpolate(self.radial_bin_centres * self.R500c) * unyt.dimensionless
        self.temperature_profile = spec_temperature_interpolate(self.radial_bin_centres * self.R500c) * unyt.keV

    @staticmethod
    def equation_hse_dlogrho_dlogr(x, rho0, rc, alpha, beta, rs, epsilon):
        """
        Function that collects the two differential terms in the equation for hydrostatic
        equilibrium mass.
        Following the notation in Vikhlinin+2006, `x` denoted the radial bin centers,
        in units of R500.
        """
        return -0.5 * (alpha + (6 * beta - alpha) * (x / rc) ** 2 / (1 + (x / rc) ** 2) + \
                       epsilon * (x / rs) ** 3 / (1 + (x / rs) ** 3))

    @staticmethod
    def equation_hse_dlogkT_dlogr(x, T0, rt, a, b, c, rcool, acool, Tmin):
        """
        Function that collects the two differential terms in the equation for hydrostatic
        equilibrium mass.
        Following the notation in Vikhlinin+2006, `x` denoted the radial bin centers,
        in units of R500.
        """
        return -a + (acool * (x / rcool) ** acool / (
                (1 + (x / rcool) ** acool) * (Tmin / T0 + (x / rcool) ** acool))) * \
               (1 - Tmin / T0) - c * (x / rt) ** b / (1 + (x / rt) ** b)

    @staticmethod
    def density_profile_model(x, rho0, rc, alpha, beta, rs, epsilon):
        return np.log10(rho0 * ((x / rc) ** (-alpha / 2) / (1 + (x / rc) ** 2) ** \
                                (3 * beta / 2 - alpha / 4)) * (1 / ((1 + (x / rs) ** 3) ** (epsilon / 6))))

    @staticmethod
    def temperature_profile_model(x, T0, rt, a, b, c, rcool, acool, Tmin):
        t = T0 * (x / rt) ** (-a) / ((1.0 + (x / rt) ** b) ** (c / b))
        x1 = (x / rcool) ** acool
        tcool = (x1 + Tmin / T0) / (x1 + 1.0)
        return t * tcool

    @staticmethod
    def residuals_temperature(free_parameters, y, x):
        T0, rt, a, b, c, rcool, acool, Tmin = free_parameters
        t = T0 * (x / rt) ** (-a) / ((1 + (x / rt) ** b) ** (c / b))
        x1 = (x / rcool) ** acool
        tcool = (x1 + Tmin / T0) / (x1 + 1)
        err = (t * tcool) - y
        return np.sum(err * err)

    @staticmethod
    def residuals_density(free_parameters, y, x):
        rho0, rc, alpha, beta, rs, epsilon = free_parameters
        err = np.log10(rho0 * ((x / rc) ** (-alpha * 0.5) / (1 + (x / rc) ** 2) ** (3 * beta / 2 - alpha / 4)) * (
                1 / ((1 + (x / rs) ** 3) ** (epsilon / 6)))) - y
        return np.sum(err * err)

    def density_fit(self, x, y):

        alpha0 = 1
        beta0 = 1
        epsilon0 = 1
        # For low mass halos, tweak the initial guess for alpha
        if self.M500c.v < 2e13:
            alpha0 = 0.2
            beta0 = 0.2
            epsilon0 = 0.2

        p0 = [100.0, 0.1, alpha0, beta0, 0.8, epsilon0]
        coeff_rho = minimize(
            self.residuals_density, p0, args=(y, x), method='L-BFGS-B',
            bounds=[
                (1e2, 1e4),
                (0.0, 10.0),
                (0.0, 10.0),
                (0.0, np.inf),
                (0.2, np.inf),
                (0.0, 5.0)
            ],
            options={'maxiter': 1e5, 'ftol': 1e-13}
        )
        return coeff_rho

    def temperature_fit(self, x, y):

        kT500 = (unyt.G * self.M500c * mean_molecular_weight * unyt.mass_proton) / (2 * self.R500c)
        kT500 = kT500.to('keV').v

        p0 = [kT500, self.R500c.v, 0.1, 3.0, 1.0, 0.1, 1.0, kT500]
        bnds = ([0.0, 0.0, -3.0, 1.0, 0.0, 0.0, 1.0e-10, 0.0],
                [np.inf, np.inf, 3.0, 5.0, 10.0, np.inf, 3.0, np.inf])

        cf1 = least_squares(self.residuals_temperature, p0, bounds=bnds, args=(y, x), max_nfev=50000)
        mod1 = self.temperature_profile_model(
            x, cf1.x[0], cf1.x[1], cf1.x[2], cf1.x[3], cf1.x[4], cf1.x[5], cf1.x[6], cf1.x[7]
        )
        xis1 = np.sum((mod1 - y) ** 2.0 / y) / len(y)

        cf2 = minimize(self.residuals_temperature, p0, args=(y, x), method='Nelder-Mead',
                       options={'maxiter': 50000, 'ftol': 1e-8})
        mod2 = self.temperature_profile_model(
            x, cf2.x[0], cf2.x[1], cf2.x[2], cf2.x[3], cf2.x[4], cf2.x[5], cf2.x[6], cf2.x[7]
        )
        xis2 = np.sum((mod2 - y) ** 2.0 / y) / len(y)

        # Assume that Nelder-Mead method works, but check xisq against bounded fit
        coeff_temp = cf2
        if xis1 < xis2:
            coeff_temp = cf1

        return coeff_temp

    def run_hse_fit(self):
        """
        This function launches the hydrostatic estimate fits for both
        density and (mass-weighted) temperature profiles.
        Returns the coefficients from the density and temperature fits
        and the total mass within the each radial bin, using the
        hydrostatic mass estimate (HSE).
        Notes:
            density_fit returns 6 parameters
            temperature_fit returns 8 parameters
            temperatures_hse wants 9 arguments (1 radial bins + 8 parameters)
            dlogkT_dlogr_hse wants 9 arguments (1 radial bins + 8 parameters)
            dlogrho_dlogr_hse wants 7 arguments (1 radial bins + 6 parameters)
        """
        cfr = self.density_fit(self.radial_bin_centres.v, np.log10(self.density_profile.v))
        cft = self.temperature_fit(self.radial_bin_centres.v, self.temperature_profile.v)

        temperatures_hse = self.temperature_profile_model(self.radial_bin_centres.v, *cft.x)
        dlogkT_dlogr_hse = self.equation_hse_dlogkT_dlogr(self.radial_bin_centres.v, *cft.x)
        dlogrho_dlogr_hse = self.equation_hse_dlogrho_dlogr(self.radial_bin_centres.v, *cfr.x)

        masses_hse = - 3.68e13 * (self.radial_bin_centres * self.R500c / unyt.Mpc) * temperatures_hse * (
                dlogrho_dlogr_hse + dlogkT_dlogr_hse) * unyt.Solar_Mass

        # Parse fitted profiles to diagnostic container
        if self.diagnostics_on:
            setattr(self.diagnostics, 'radial_bin_centres_hse', self.radial_bin_centres)
            setattr(self.diagnostics, 'temperature_profile_hse', temperatures_hse * unyt.keV)
            setattr(self.diagnostics, 'cumulative_mass_hse', masses_hse)

            gas_density = 10 ** self.density_profile_model(self.radial_bin_centres.v, *cfr.x)
            setattr(self.diagnostics, 'density_profile_hse', gas_density)

            # Compute density profile from cumulative mass
            mass_in_shell = masses_hse[1:] - masses_hse[:-1]
            volume_in_shell = 4 / 3 * np.pi * self.R500c ** 3 * (
                    self.radial_bin_centres[1:] ** 3 - self.radial_bin_centres[:-1] ** 3)
            density_in_shell = mass_in_shell / volume_in_shell / self.rho_crit
            mass_interpolate = interp1d(self.radial_bin_edges[1:-1], density_in_shell, fill_value='extrapolate')
            total_density = mass_interpolate(self.radial_bin_centres)
            setattr(self.diagnostics, 'total_density_profile_hse', total_density)

        return cfr, cft, masses_hse

    def interpolate_hse(self):

        _, _, masses_hse = self.run_hse_fit()
        mass_interpolate = interp1d(self.radial_bin_centres * self.R500c, masses_hse,
                                    kind='linear', fill_value='extrapolate')
        densities_hse = (3 * masses_hse) / (4 * np.pi * (self.radial_bin_centres * self.R500c) ** 3) / self.rho_crit
        density_interpolate = interp1d(densities_hse, self.radial_bin_centres * self.R500c,
                                       kind='linear', fill_value='extrapolate')

        self.R200hse = density_interpolate(200) * unyt.Mpc
        self.R500hse = density_interpolate(500) * unyt.Mpc
        self.R2500hse = density_interpolate(2500) * unyt.Mpc

        self.M200hse = mass_interpolate(self.R200hse) * unyt.Solar_Mass
        self.M500hse = mass_interpolate(self.R500hse) * unyt.Solar_Mass
        self.M2500hse = mass_interpolate(self.R2500hse) * unyt.Solar_Mass

        self.ne200hse = (3 * self.M200hse * fbary / (
                4 * np.pi * self.R200hse ** 3 * unyt.mass_proton * mean_molecular_weight)).to('1/cm**3')
        self.ne500hse = (3 * self.M500hse * fbary / (
                4 * np.pi * self.R500hse ** 3 * unyt.mass_proton * mean_molecular_weight)).to('1/cm**3')
        self.ne2500hse = (3 * self.M2500hse * fbary / (
                4 * np.pi * self.R2500hse ** 3 * unyt.mass_proton * mean_molecular_weight)).to('1/cm**3')

        self.kBT200hse = (unyt.G * mean_molecular_weight * self.M200hse * unyt.mass_proton / self.R200hse / 2).to('keV')
        self.kBT500hse = (unyt.G * mean_molecular_weight * self.M500hse * unyt.mass_proton / self.R500hse / 2).to('keV')
        self.kBT2500hse = (unyt.G * mean_molecular_weight * self.M2500hse * unyt.mass_proton / self.R2500hse / 2).to(
            'keV')

        self.P200hse = (200 * fbary * self.rho_crit * unyt.G * self.M200hse / self.R200hse / 2).to('keV/cm**3')
        self.P500hse = (500 * fbary * self.rho_crit * unyt.G * self.M500hse / self.R500hse / 2).to('keV/cm**3')
        self.P2500hse = (2500 * fbary * self.rho_crit * unyt.G * self.M2500hse / self.R2500hse / 2).to('keV/cm**3')

        self.K200hse = (self.kBT200hse / (
                3 * self.M200hse * fbary / (4 * np.pi * self.R200hse ** 3 * unyt.mass_proton)) ** (2 / 3)).to(
            'keV*cm**2')
        self.K500hse = (self.kBT500hse / (
                3 * self.M500hse * fbary / (4 * np.pi * self.R500hse ** 3 * unyt.mass_proton)) ** (2 / 3)).to(
            'keV*cm**2')
        self.K2500hse = (self.kBT2500hse / (
                3 * self.M2500hse * fbary / (4 * np.pi * self.R2500hse ** 3 * unyt.mass_proton)) ** (2 / 3)).to(
            'keV*cm**2')

        # Hydrostatic bias
        self.b200hse = 1 - self.M200hse / self.M200c
        self.b500hse = 1 - self.M500hse / self.M500c
        self.b2500hse = 1 - self.M2500hse / self.M2500c

        # Transfer bias info to the diagnostics instance
        if self.diagnostics_on:
            setattr(self.diagnostics, 'b200hse', self.b200hse)
            setattr(self.diagnostics, 'b500hse', self.b500hse)
            setattr(self.diagnostics, 'b2500hse', self.b2500hse)
            setattr(self.diagnostics, 'R500hse', self.R500hse)

    def plot_diagnostics(self):
        if not self.diagnostics_on:
            raise ValueError((
                "Diagnostic plots cannot be produced as diagnostics class instance not initialised. "
                "Select `diagnostics_on: bool = True` when initialising HydrostaticEstimator."
            ))
        self.diagnostics.plot_all()


if __name__ == "__main__":
    zoom_choice = [z for z in zooms_register if "VR2915_-8res_MinimumDistance_fixedAGNdT8.5_" in z.run_name][0]
    print(zoom_choice.run_name)

    hse_test = HydrostaticEstimator(zoom_choice, diagnostics_on=True)

    print(f'R500c = {hse_test.R500c:.3E}')
    print(f'R500hse = {hse_test.R500hse:.3E}')
    print()
    print(f'M500c = {hse_test.M500c:.3E}')
    print(f'M500hse = {hse_test.M500hse:.3E}')
    print(f"Mass bias (200hse) = {hse_test.b200hse:.3f}")
    print(f"Mass bias (500hse) = {hse_test.b500hse:.3f}")
    print(f"Mass bias (2500hse) = {hse_test.b2500hse:.3f}")
    print()
    print(f'P500hse = {hse_test.P500hse:.3E}')
    print(f'kBT500hse = {hse_test.kBT500hse:.3E}')
    print(f'K500hse = {hse_test.K500hse:.3E}')

    hse_test.plot_diagnostics()
