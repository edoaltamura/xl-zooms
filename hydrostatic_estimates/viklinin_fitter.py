import sys
import os
import unyt
import numpy as np
from typing import Tuple
import h5py as h5
import swiftsimio as sw
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares, curve_fit


# Make the register backend visible to the script
sys.path.append("../zooms")
sys.path.append("../observational_data")

from register import zooms_register, Zoom, Tcut_halogas, name_list
from convergence_radius import convergence_radius
# import observational_data as obs
# import scaling_utils as utils
# import scaling_style as style

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass


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


class HydrostaticEstimator:

    def __init__(self, zoom: Zoom, excise_core: bool = True, using_mcmc: bool = False):
        self.zoom = zoom
        self.using_mcmc = using_mcmc
        self.excise_core = excise_core
        self.load_zoom_profiles()

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
        rho_crit = unyt.unyt_quantity(
            data.metadata.cosmology_raw['Critical density [internal units]'],
            unitMass / unitLength ** 3
        )

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
        gas_temperatures = data.gas.temperatures[index]
        gas_mass_weighted_temperatures = gas_temperatures * gas_masses

        # Set bounds for the radial profiles
        radius_bounds = [0.15, 5]
        if not self.excise_core:
            # Compute convergence radius and set as inner limit
            gas_convergence_radius = convergence_radius(
                deltaR[index],
                gas_masses.to('Msun'),
                rho_crit.to('Msun/Mpc**3')
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

        # Compute the radial gas density profile
        volume_shell = (4. * np.pi / 3.) * (self.R500c ** 3) * ((bin_edges[1:]) ** 3 - (bin_edges[:-1]) ** 3)
        self.density_profile = mass_weights / volume_shell / rho_crit
        assert self.density_profile.units == unyt.dimensionless

        # Compute the radial mass-weighted temperature profile
        hist, _ = histogram_unyt(radial_distance_scaled, bins=lbins, weights=gas_mass_weighted_temperatures)
        hist /= mass_weights
        self.temperature_profile = (hist * unyt.boltzmann_constant).to('keV')

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

        p0 = [100.0, 0.1, 1.0, 1.0, 0.8 * self.R500, 1.0]
        coeff_rho = minimize(
            self.residuals_density, p0, args=(y, x), method='L-BFGS-B',
            bounds=[
                (1.0e2, 1.0e4),
                (0.0, 10.0),
                (0.0, 10.0),
                (0.0, np.inf),
                (0.2 * self.R500, np.inf),
                (0.0, 5.0)
            ],
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        return coeff_rho

    def temperature_fit(self, x, y):

        kT500 = (unyt.G * self.M500 * mean_molecular_weight * unyt.mass_proton) / (2 * self.R500)
        kT500 = kT500.to('keV')

        p0 = [kT500, self.R500, 0.1, 3.0, 1.0, 0.1, 1.0, kT500]
        bnds = ([0.0, 0.0, -3.0, 1.0, 0.0, 0.0, 1.0e-10, 0.0],
                [np.inf, np.inf, 3.0, 5.0, 10.0, np.inf, 3.0, np.inf])

        cf1 = least_squares(self.residuals_temperature, p0, bounds=bnds, args=(y, x), max_nfev=2000)
        mod1 = self.temperature_profile_model(
            x, cf1.x[0], cf1.x[1], cf1.x[2], cf1.x[3], cf1.x[4], cf1.x[5], cf1.x[6], cf1.x[7]
        )
        xis1 = np.sum((mod1 - y) ** 2.0 / y) / len(y)

        cf2 = minimize(self.residuals_temperature, p0, args=(y, x), method='Nelder-Mead',
                       options={'maxiter': 2000, 'ftol': 1e-5})
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
        cfr = self.density_fit(self.radial_bin_centres, np.log10(self.density_profile))
        cft = self.temperature_fit(self.radial_bin_centres, self.temperature_profile)

        temperatures_hse = self.temperature_profile_model(
            self.radial_bin_centres,
            cft.x[0], cft.x[1], cft.x[2], cft.x[3], cft.x[4], cft.x[5], cft.x[6], cft.x[7]
        )
        dT_hse = self.equation_of_state_dlogkT_dlogr(
            self.radial_bin_centres,
            cft.x[0], cft.x[1], cft.x[2], cft.x[3], cft.x[4], cft.x[5], cft.x[6], cft.x[7]
        )
        drho_hse = self.equation_of_state_dlogrho_dlogr(
            self.radial_bin_centres,
            cfr.x[0], cfr.x[1], cfr.x[2], cfr.x[3], cfr.x[4], cfr.x[5]
        )
        print(temperatures_hse)
        masses_hse = - 3.68e13 * (self.radial_bin_centres * self.R500c) * temperatures_hse * (drho_hse + dT_hse)

        return cfr, cft, masses_hse


if __name__ == "__main__":
    hse_test = HydrostaticEstimator(zooms_register[0])
    print(hse_test.run_hse_fit())