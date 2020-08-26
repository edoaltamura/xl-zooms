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
    radial_distances_sorted = radial_distances[sort_rule][2:]
    particle_masses_sorted = particle_masses[sort_rule][2:]
    number_particles = np.arange(len(particle_masses))[2:]

    # Compute the RHS of the equation
    sphere_volume = 4 / 3 * np.pi * radial_distances_sorted ** 3
    mean_rho = np.cumsum(particle_masses_sorted) / sphere_volume
    result = np.sqrt(200) / 8 * number_particles / np.log(number_particles) * np.sqrt(rho_crit / mean_rho)

    # Find solutions by minimising the root function
    root_idx = np.abs(result - alpha).argmin()
    convergence_root = (radial_distances_sorted[root_idx])

    # Convergence radius: assume t_relax=t_200
    radSort = np.sort(radial_distances)
    volSort = (4. * np.pi / 3.) * (radSort ** 3)
    intNumber = np.arange(radSort.size)
    intMass = particle_masses[0] * intNumber
    intRho = intMass / volSort
    convRatio = (np.sqrt(200.) / 8.) * (intNumber / np.log(intNumber)) * np.sqrt(rho_crit / intRho)
    # Delete first two entries (entry 0 has zero mass, entry 1 has zero log(N))
    radSort = radSort[2:]
    convRatio = convRatio[2:]
    index = np.where(convRatio >= 1)[0][0]
    rConvParent = radSort[index]
    print(convergence_root, rConvParent, root_idx, index)

    return convergence_root * unyt.Mpc