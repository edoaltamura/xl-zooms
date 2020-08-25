from typing import Tuple
import numpy as np


def convergence_radius(radial_distances: np.ndarray, particle_masses: np.ndarray, rho_crit: float) -> Tuple[float, float]:
    """
    Function that computes the inner numerical convergence radius for point particles.
    Calculation based on Power at al. (2003).
    See https://ui.adsabs.harvard.edu/abs/2003MNRAS.338...14P/abstract.

    Equation to satisfy:
    \\alpha = \\frac{\\sqrt{200}}{8} \\frac{N}{\\ln N} \\left(\\frac{\\rho(r)}{\\rho_c} \\right)^{-1/2}

    \\alpha can be 1 or 0.6
    N is the number of particles enclosed within the sphere
    \\rho(r) is the mean mass density within the enclosed sphere of radius r

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
    numerical_tolerance = 1e-5
    alphas = [0.6, 1.]
    inner_radii = []

    assert len(radial_distances) == len(particle_masses)

    # Sort particle radial distance from the centre of the halo
    sort_rule = radial_distances.argsort()
    radial_distances = radial_distances[sort_rule]
    particle_masses = particle_masses[sort_rule]
    counter = 2
    for alpha in alphas:
        while True:

            sphere_volume = 3/4 * np.pi * radial_distances[counter-1] ** 3
            mean_rho = np.cumsum(particle_masses[:counter]) / sphere_volume
            result = np.sqrt(200)/8 * counter/np.log(counter) * (mean_rho/rho_crit) ** (-0.5)

            if np.abs(result-alpha) / np.max([result, alpha]) < numerical_tolerance:
                break
            elif mean_rho/rho_crit < 200:
                raise RuntimeError("Convergence might be outside the virial radius.")
            else:
                counter += 1
                continue

        inner_radii.append(radial_distances[counter])

    return tuple(inner_radii)