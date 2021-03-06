import numpy as np
import unyt

alpha = 1.

def convergence_radius(
        radial_distances: unyt.array.unyt_array,
        particle_masses: unyt.array.unyt_array,
        rho_crit: unyt.array.unyt_quantity
) -> unyt.array.unyt_quantity:
    """
    Function that computes the inner numerical convergence radius for point particles.
    Calculation based on Power at al. (2003).
    See https://ui.adsabs.harvard.edu/abs/2003MNRAS.338...14P/abstract.

    Equation to satisfy:
    \\alpha = \\frac{\\sqrt{200}}{8} \\frac{N}{\\ln N} \\left(\\frac{\\rho(r)}{\\rho_c} \\right)^{-1/2}

    \\alpha is a global variable (default = 1)
    N is the number of particles enclosed within the sphere
    \\rho(r) is the mean mass density within the enclosed sphere of radius r

    :param radial_distances: unyt.array.unyt_array
        The radial distances array of the particles (not necessarily sorted).
    :param particle_masses: unyt.array.unyt_array
        The radial distances array of the particles (not necessarily sorted).
    :param rho_crit: unyt.array.unyt_quantity
        The critical density of the Universe, as reported in the simulation.
    :return: unyt.array.unyt_array
        Returns the inner convergence radius for the specified value of alpha.
    """

    # Check arguments are in the correct format
    assert len(radial_distances) == len(particle_masses)
    assert radial_distances.units == unyt.Mpc
    assert particle_masses.units == unyt.Solar_Mass
    assert rho_crit.units == unyt.Solar_Mass / unyt.Mpc ** 3

    # Sort particle radial distance from the centre of the halo
    sort_rule = radial_distances.argsort()
    radial_distances_sorted = radial_distances[sort_rule][2:]
    particle_masses_sorted = particle_masses[sort_rule][2:]
    number_particles = np.arange(len(particle_masses))[2:]

    # Compute the RHS of the equation
    sphere_volume = 4 / 3 * np.pi * radial_distances_sorted ** 3
    rho_mean = np.cumsum(particle_masses_sorted) / sphere_volume
    result = np.sqrt(200) / 8 * number_particles / np.log(number_particles) * np.sqrt(rho_crit / rho_mean)

    # Find solutions by minimising the root function
    root_idx = np.abs(result - alpha).argmin()
    convergence_root = radial_distances_sorted[root_idx]

    return convergence_root