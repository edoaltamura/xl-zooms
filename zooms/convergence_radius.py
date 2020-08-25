from typing import Tuple
import numpy as np
from scipy import interpolate


def convergence_radius(radial_distances: np.ndarray, particle_masses: np.ndarray, r200c: float, rho_crit: float) -> Tuple[float, float]:
    """
    Function that computes the inner numerical convergence radius for point particles.
    Calculation based on Power at al. (2003).
    See https://ui.adsabs.harvard.edu/abs/2003MNRAS.338...14P/abstract.

    Equation to satisfy:
    \\alpha = \\frac{\\sqrt{200}}{8} \\frac{N}{\\ln N} \\left(\\frac{\\rho(r)}{\\rho_c} \\right)^{-1/2}

    \\alpha can be 1 or 0.6
    N is the number of particles enclosed within the sphere
    \\rho(r) is the mean mass density within the enclosed sphere of radius r

    :param r200c: 
    :param radial_distances: np.ndarray
        The radial distances array of the particles (not necessarily sorted).
    :param particle_masses: float or np.ndarray
        The radial distances array of the particles (not necessarily sorted).
    :param rho_crit: float
        The critical density of the Universe, as reported in the simulation.
    :return: tuple(float, float)
        Returns the inner convergence radius for the values of:
            - \\alpha = 0.6
            - \\alpha = 1
    """

    # Set-up
    numerical_tolerance = 1e-3
    alphas = [0.6, 1.]
    inner_radii = []

    assert len(radial_distances) == len(particle_masses)

    # Sort particle radial distance from the centre of the halo
    sort_rule = radial_distances.argsort()
    radial_distances_sorted = radial_distances[sort_rule][1:] / r200c
    particle_masses_sorted = particle_masses[sort_rule][1:]
    number_particles = np.linspace(1, len(particle_masses), len(particle_masses), dtype=np.int)[1:]

    # Compute the RHS of the equation
    sphere_volume = 3 / 4 * np.pi * radial_distances_sorted ** 3
    mean_rho = np.cumsum(particle_masses_sorted) / sphere_volume
    result = np.sqrt(200) / 8 * number_particles / np.log(number_particles) * (mean_rho / rho_crit) ** (-0.5)

    # Find solutions by interpolation
    for alpha in alphas:
        solution_idx = np.where(np.abs(result - alpha) < numerical_tolerance)[0]
        smooth_function = interpolate.interp1d(result[solution_idx], radial_distances_sorted[solution_idx])
        inner_radii.append(smooth_function(alpha))

        # elif mean_rho / rho_crit < 200:
        #     raise RuntimeError("Convergence might be outside the virial radius.")

    return tuple(inner_radii)