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

# import scaling_utils as utils
# import scaling_style as style

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

cosmology = obs.Observations().cosmo_model
fbary = cosmology.Ob0 / cosmology.Om0  # Cosmic baryon fraction
mean_molecular_weight = 0.5954


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
    # This class is to be used for debugging purposes and
    # the allowed attributes are specified in __slots__.
    __slots__ = (
        'profile_type',
        'zoom',
        'output_directory',
        'radial_bin_centres_input',
        'temperature_profile_input',
        'cumulative_mass_input',
        'density_profile_input',
        'radial_bin_centres_hse',
        'temperature_profile_hse',
        'cumulative_mass_hse',
        'density_profile_hse',
    )

    def __init__(self, zoom: Zoom):
        self.zoom = zoom
        self.total_mass_profiles()

    def total_mass_profiles(self):

        # Read in halo properties from catalog
        with h5.File(self.zoom.catalog_file, 'r') as h5file:
            M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
            R500c = unyt.unyt_quantity(h5file['/SO_R_500_rhocrit'][0], unyt.Mpc)
            XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc)
            YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc)
            ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc)

        # Read in gas particles and parse densities and temperatures
        mask = sw.mask(self.zoom.snapshot_file, spatial_only=False)
        region = [[XPotMin - 1.5 * R500c, XPotMin + 1.5 * R500c],
                  [YPotMin - 1.5 * R500c, YPotMin + 1.5 * R500c],
                  [ZPotMin - 1.5 * R500c, ZPotMin + 1.5 * R500c]]
        mask.constrain_spatial(region)
        mask.constrain_mask(
            "gas", "temperatures",
            Tcut_halogas * mask.units.temperature,
            1.e12 * mask.units.temperature
        )
        data = sw.load(self.zoom.snapshot_file, mask=mask)

        # Set bounds for the radial profiles
        radius_bounds = [0.15, 1.5]

        # Select hot gas within sphere and without core
        deltaX = data.gas.coordinates[:, 0] - XPotMin
        deltaY = data.gas.coordinates[:, 1] - YPotMin
        deltaZ = data.gas.coordinates[:, 2] - ZPotMin
        deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) / R500c

        # Keep only particles inside 1.5 R500crit
        index = np.where(
            (deltaR > radius_bounds[0]) &
            (deltaR < radius_bounds[1])
        )[0]

        radial_distance = deltaR[index]
        masses = data.gas.masses[index]

        # Select DM within sphere and without core
        deltaX = data.dark_matter.coordinates[:, 0] - XPotMin
        deltaY = data.dark_matter.coordinates[:, 1] - YPotMin
        deltaZ = data.dark_matter.coordinates[:, 2] - ZPotMin
        deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) / R500c

        # Keep only particles inside 1.5 R500crit
        index = np.where(
            (deltaR > radius_bounds[0]) &
            (deltaR < radius_bounds[1])
        )[0]

        radial_distance = np.append(radial_distance, deltaR[index])
        masses = np.append(masses, data.dark_matter.masses[index])

        # Select stars within sphere and without core
        deltaX = data.stars.coordinates[:, 0] - XPotMin
        deltaY = data.stars.coordinates[:, 1] - YPotMin
        deltaZ = data.stars.coordinates[:, 2] - ZPotMin
        deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) / R500c

        # Keep only particles inside 1.5 R500crit
        index = np.where(
            (deltaR > radius_bounds[0]) &
            (deltaR < radius_bounds[1])
        )[0]

        radial_distance = np.append(radial_distance, deltaR[index])
        masses = np.append(masses, data.stars.masses[index])

        lbins = np.logspace(
            np.log10(radius_bounds[0]), np.log10(radius_bounds[1]), 501
        ) * radial_distance.units

        mass_weights, bin_edges = histogram_unyt(radial_distance, bins=lbins, weights=masses)

        # Replace zeros with Nans
        mass_weights[mass_weights == 0] = np.nan

        self.radial_bin_centres_input = np.sqrt(bin_edges[1:] * bin_edges[:-1])
        self.cumulative_mass_input = cumsum_unyt(mass_weights)
        shell_volume = (4 / 3 * np.pi) * R500c ** 3 * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
        critical_density = data.metadata.cosmology.critical_density(data.metadata.z)
        self.density_profile_input = mass_weights / shell_volume / critical_density

    def plot_all(self):
        fields = [
            'temperature_profile',
            'cumulative_mass',
            'density_profile'
        ]
        labels = [
            r'$k_{\rm B}T$ [keV]',
            r'$M(<R)$ [M$_\odot$]',
            r'$\rho / \rho_{\rm crit}$'
        ]
        for i, (field, label) in enumerate(zip(fields, labels)):
            print(f"({i + 1}/{len(fields)}) Generating diagnostic plot: {field}.")
            self.plot_profile(
                field_name=field,
                ylabel=label,
                filename=f"diagnostic_{field}.png"
            )

    def plot_profile(self, field_name: str = 'radial_bin_centres',
                     ylabel: str = 'y', filename: str = 'diagnostics.png'):

        fig, (ax, ax_residual) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(4, 5),
            dpi=300,
            sharex=True,
            gridspec_kw={'height_ratios': [3, 1]}
        )

        x_input = self.radial_bin_centres_input
        x_hse = self.radial_bin_centres_hse

        ax.plot(x_input, getattr(self, field_name + '_input'), label="True data")
        ax.plot(x_hse, getattr(self, field_name + '_hse'), label=f"Vikhlinin HSE fit from {self.profile_type} data")
        ax_residual.plot(x_hse, (getattr(self, field_name + '_hse') - getattr(self, field_name + '_input')) / \
                         getattr(self, field_name + '_input'))

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(ylabel)
        ax_residual.set_ylabel(r"$\Delta$" + ylabel)
        ax_residual.set_xlabel(r"$R\ /\ R_{\rm 500c\ (true)}$")
        ax.legend(loc="upper right")
        fig.tight_layout()
        plt.title(self.zoom.run_name)
        plt.savefig(f"{self.output_directory}/{filename}")
        plt.show()
        plt.close()


class HydrostaticEstimator:

    def __init__(self, zoom: Zoom, excise_core: bool = True, profile_type: str = 'true',
                 using_mcmc: bool = False, spec_fit_data: dict = None):
        self.zoom = zoom
        self.using_mcmc = using_mcmc
        self.excise_core = excise_core
        self.profile_type = profile_type

        # Initialise an HydrostaticDiagnostic instance
        # Parse to this objects all profiles and quantities for external checks
        self.diagnostics = HydrostaticDiagnostic(zoom)

        if profile_type.lower() == 'true':
            self.load_zoom_profiles()
        elif profile_type.lower() == 'spec' or 'xray' in profile_type.lower():
            assert spec_fit_data, "Spec output data not detected."
            self.load_xray_profiles(spec_fit_data)

        # Parse fitted profiles to diagnostic container
        setattr(self.diagnostics, 'profile_type', self.profile_type)
        setattr(self.diagnostics, 'output_directory', self.zoom.output_directory)
        setattr(self.diagnostics, 'temperature_profile_input', self.temperature_profile)

        self.interpolate_hse()

    @classmethod
    def from_data_paths(cls, catalog_file: str, snapshot_file: str,
                        excise_core: bool = True, profile_type: str = 'true',
                        using_mcmc: bool = False, spec_fit_data: dict = None):
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
                   using_mcmc=using_mcmc, spec_fit_data=spec_fit_data)

    def load_zoom_profiles(self):
        # Read in halo properties from catalog
        with h5.File(self.zoom.catalog_file, 'r') as h5file:
            self.M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
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
            np.log10(radius_bounds[0]), np.log10(radius_bounds[1]), 501
        ) * radial_distance_scaled.units

        mass_weights, bin_edges = histogram_unyt(radial_distance_scaled, bins=lbins, weights=gas_masses)

        # Replace zeros with Nans
        mass_weights[mass_weights == 0] = np.nan

        # Set the radial bins as object attribute
        self.radial_bin_centres = np.sqrt(bin_edges[1:] * bin_edges[:-1])
        self.mass_profile = mass_weights

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
            self.M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
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
            np.log10(radius_bounds[0]), np.log10(radius_bounds[1]), 1001
        ) * self.R500c

        mass_weights, bin_edges = histogram_unyt(radial_distance_scaled, bins=lbins, weights=gas_masses)

        # Replace zeros with Nans
        mass_weights[mass_weights == 0] = np.nan

        self.radial_bin_centres = 10.0 ** (0.5 * np.log10(lbins[1:] * lbins[:-1]))
        self.mass_profile = mass_weights

        # Cut ends of the radial bins to interpolate, since they might be
        # outside the spec_fit_data['Rspec'] range
        # Prevents ValueError: A value in x_new is below the interpolation range.
        radial_bins_intersect = np.where(
            (self.radial_bin_centres > spec_fit_data['Rspec'].min()) &
            (self.radial_bin_centres < spec_fit_data['Rspec'].max())
        )[0]
        self.radial_bin_centres = self.radial_bin_centres[radial_bins_intersect] * radial_distance_scaled.units

        # Compute the radial gas density profile
        self.density_profile = spec_density_interpolate(self.radial_bin_centres) * unyt.dimensionless
        self.temperature_profile = spec_temperature_interpolate(self.radial_bin_centres) * unyt.keV

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
        return np.log10(rho0 * ((x / rc) ** (-alpha / 2.0) / (1.0 + (x / rc) ** 2.0) ** \
                                (3.0 * beta / 2.0 - alpha / 4.0)) * (1.0 / ((1.0 + (x / rs) ** 3.0) ** \
                                                                            (epsilon / 6.0))))

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

        p0 = [100.0, 0.1, 1.0, 1.0, 0.8 * self.R500c.v, 1.0]
        coeff_rho = minimize(
            self.residuals_density, p0, args=(y, x), method='L-BFGS-B',
            bounds=[
                (1.0e2, 1.0e4),
                (0.0, 10.0),
                (0.0, 10.0),
                (0.0, np.inf),
                (0.2 * self.R500c.v, np.inf),
                (0.0, 5.0)
            ],
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        return coeff_rho

    def temperature_fit(self, x, y):

        kT500 = (unyt.G * self.M500c * mean_molecular_weight * unyt.mass_proton) / (2 * self.R500c)
        kT500 = kT500.to('keV').v

        p0 = [kT500, self.R500c.v, 0.1, 3.0, 1.0, 0.1, 1.0, kT500]
        bnds = ([0.0, 0.0, -3.0, 1.0, 0.0, 0.0, 1.0e-10, 0.0],
                [np.inf, np.inf, 3.0, 5.0, 10.0, np.inf, 3.0, np.inf])

        cf1 = least_squares(self.residuals_temperature, p0, bounds=bnds, args=(y, x), max_nfev=10000)
        mod1 = self.temperature_profile_model(
            x, cf1.x[0], cf1.x[1], cf1.x[2], cf1.x[3], cf1.x[4], cf1.x[5], cf1.x[6], cf1.x[7]
        )
        xis1 = np.sum((mod1 - y) ** 2.0 / y) / len(y)

        cf2 = minimize(self.residuals_temperature, p0, args=(y, x), method='Nelder-Mead',
                       options={'maxiter': 10000, 'ftol': 1e-5})
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
        """
        cfr = self.density_fit(self.radial_bin_centres.v, np.log10(self.density_profile.v))
        cft = self.temperature_fit(self.radial_bin_centres.v, self.temperature_profile.v)

        temperatures_hse = self.temperature_profile_model(
            self.radial_bin_centres,
            cft.x[0], cft.x[1], cft.x[2], cft.x[3], cft.x[4], cft.x[5], cft.x[6], cft.x[7]
        )
        dT_hse = self.equation_hse_dlogkT_dlogr(
            self.radial_bin_centres,
            cft.x[0], cft.x[1], cft.x[2], cft.x[3], cft.x[4], cft.x[5], cft.x[6], cft.x[7]
        )
        drho_hse = self.equation_hse_dlogrho_dlogr(
            self.radial_bin_centres,
            cfr.x[0], cfr.x[1], cfr.x[2], cfr.x[3], cfr.x[4], cfr.x[5]
        )

        masses_hse = - 3.68e13 * (self.radial_bin_centres * self.R500c / unyt.Mpc) * temperatures_hse * (
                drho_hse + dT_hse) * unyt.Solar_Mass

        # Parse fitted profiles to diagnostic container
        setattr(self.diagnostics, 'radial_bin_centres_hse', self.radial_bin_centres)
        setattr(self.diagnostics, 'temperature_profile_hse', temperatures_hse * unyt.keV)
        setattr(self.diagnostics, 'cumulative_mass_hse', masses_hse)
        mass_in_shell = masses_hse[1:] - masses_hse[:-1]
        volume_in_shell = 4 / 3 * np.pi * (self.radial_bin_centres[1:] ** 3 - self.radial_bin_centres[:-1] ** 3)
        density_interpolate = interp1d(
            mass_in_shell / volume_in_shell / self.rho_crit,
            self.radial_bin_centres[1:] - self.radial_bin_centres[:-1],
            kind='linear', fill_value="extrapolate"
        )
        setattr(self.diagnostics, 'density_profile_hse', density_interpolate(self.radial_bin_centres))

        return cfr, cft, masses_hse

    def interpolate_hse(self):

        _, _, masses_hse = self.run_hse_fit()
        mass_interpolate = interp1d(self.radial_bin_centres, masses_hse, kind='linear')
        densities_hse = (3 * masses_hse) / (4 * np.pi * self.radial_bin_centres ** 3) / self.rho_crit
        density_interpolate = interp1d(densities_hse, self.radial_bin_centres, kind='linear')

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

    def plot_diagnostics(self):
        self.diagnostics.plot_all()


if __name__ == "__main__":
    zoom_choice = [z for z in zooms_register if "VR3032_-8res" in z.run_name][0]
    print(zoom_choice.run_name)

    hse_test = HydrostaticEstimator(zoom_choice)

    print('R500c =', hse_test.R500c)
    print('M500c =', hse_test.M500c)
    print('R500hse =', hse_test.R500hse)
    print('M500hse =', hse_test.M500hse)
    print('P500hse =', hse_test.P500hse)
    print('kBT500hse =', hse_test.kBT500hse)
    print('K500hse =', hse_test.K500hse)

    hse_test.plot_diagnostics()
