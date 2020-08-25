import numpy as np
import unyt

alpha = 1.

def convergence_radius(radial_distances: np.ndarray, particle_masses: np.ndarray, rho_crit: float) -> unyt.array.unyt_array:
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
    :return: unyt.array.unyt_array
        Returns the inner convergence radius for the values of:
            - \\alpha = 0.6
            - \\alpha = 1
    """

    assert len(radial_distances) == len(particle_masses)

    # Sort particle radial distance from the centre of the halo
    sort_rule = radial_distances.argsort()
    radial_distances_sorted = radial_distances[sort_rule][1:]
    particle_masses_sorted = particle_masses[sort_rule][1:]

    # Begin counting particles from 2 (2-body relaxation)
    number_particles = np.linspace(2, len(particle_masses), len(particle_masses) - 1, dtype=np.int)

    # Compute the RHS of the equation
    sphere_volume = 3 / 4 * np.pi * radial_distances_sorted ** 3
    mean_rho = np.cumsum(particle_masses_sorted) / sphere_volume
    result = np.sqrt(200) / 8 * number_particles / np.log(number_particles) * (mean_rho / rho_crit) ** (-0.5)

    # Find solutions by minimising the root function
    root_idx = (result - alpha).argmin()
    convergence_root = (radial_distances_sorted[root_idx])

    return convergence_root * unyt.Mpc