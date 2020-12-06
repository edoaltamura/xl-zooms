import unyt
import numpy as np
from typing import Tuple
from multiprocessing import Pool, cpu_count
import h5py as h5
import swiftsimio as sw
import pandas as pd
import matplotlib.pyplot as plt

from register import zooms_register, Zoom, Tcut_halogas, name_list

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

# Constants
bins = 20
radius_bounds = [0.1, 3]  # In units of R500crit
fbary = 0.15741  # Cosmic baryon fraction


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def profile_3d_single_halo(path_to_snap: str, path_to_catalogue: str, weights: str) -> Tuple[np.ndarray]:
    # Read in halo properties
    with h5.File(f'{path_to_catalogue}', 'r') as h5file:
        XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc)
        YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc)
        ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc)
        M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
        R500c = unyt.unyt_quantity(h5file['/SO_R_500_rhocrit'][0], unyt.Mpc)

    # Read in gas particles
    mask = sw.mask(f'{path_to_snap}', spatial_only=False)
    region = [[XPotMin - R500c, XPotMin + R500c],
              [YPotMin - R500c, YPotMin + R500c],
              [ZPotMin - R500c, ZPotMin + R500c]]
    mask.constrain_spatial(region)
    mask.constrain_mask("gas", "temperatures", Tcut_halogas * mask.units.temperature, 1.e12 * mask.units.temperature)
    data = sw.load(f'{path_to_snap}', mask=mask)
    posGas = data.gas.coordinates

    # Select hot gas within sphere
    deltaX = posGas[:, 0] - XPotMin
    deltaY = posGas[:, 1] - YPotMin
    deltaZ = posGas[:, 2] - ZPotMin
    deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2)

    # Calculate particle mass and rho_crit
    unitLength = data.metadata.units.length
    unitMass = data.metadata.units.mass
    rho_crit = unyt.unyt_quantity(
        data.metadata.cosmology_raw['Critical density [internal units]'],
        unitMass / unitLength ** 3
    )
    dm_masses = data.dark_matter.masses.to('Msun')
    zoom_mass_resolution = dm_masses[0]

    # Since useful for different applications, attach the electron number density dataset
    data.gas.electron_number_densities = (data.gas.densities.to('Msun/Mpc**3') / 1.14 / unyt.mass_hydrogen)
    data.gas.mass_weighted_temperatures = data.gas.masses.to('Msun') * data.gas.temperatures
    # Construct bins and compute density profile
    lbins = np.logspace(np.log10(radius_bounds[0]), np.log10(radius_bounds[1]), bins)

    # Allocate weights
    if weights.lower() == 'gas_mass':
        weights_field = data.gas.masses.to('Msun')
        hist, bin_edges = np.histogram(deltaR / R500c, bins=lbins, weights=weights_field.value)
        hist *= weights_field.units

    if weights.lower() == 'gas_mass_cumulative':
        weights_field = data.gas.masses.to('Msun')
        hist, bin_edges = np.histogram(deltaR / R500c, bins=lbins, weights=weights_field.value)
        hist = np.cumsum(hist)
        hist *= weights_field.units

    elif weights.lower() == 'gas_density':
        weights_field = data.gas.densities.to('Msun/Mpc**3')
        hist, bin_edges = np.histogram(deltaR / R500c, bins=lbins, weights=weights_field.value)
        hist *= weights_field.units

    elif weights.lower() == 'dm_density':
        weights_field = dm_masses
        hist, bin_edges = np.histogram(deltaR / R500c, bins=lbins, weights=weights_field.value)
        volume_shell = (4. * np.pi / 3.) * (R500c ** 3) * ((bin_edges[1:]) ** 3 - (bin_edges[:-1]) ** 3)
        hist = hist * dm_masses.units / volume_shell / rho_crit
        # Correct for the universal baryon fraction
        hist /= (1 - fbary)

    elif weights.lower() == 'mass_weighted_temps':
        weights_field = data.gas.mass_weighted_temperatures
        hist, bin_edges = np.histogram(deltaR / R500c, bins=lbins, weights=weights_field.value)
        mass_hist, _ = np.histogram(deltaR / R500c, bins=lbins, weights=data.gas.masses.to('Msun'))
        hist = hist / mass_hist
        hist *= unyt.K

    elif weights.lower() == 'mass_weighted_temps_kev':
        weights_field = data.gas.mass_weighted_temperatures
        hist, bin_edges = np.histogram(deltaR / R500c, bins=lbins, weights=weights_field.value)
        mass_hist, _ = np.histogram(deltaR / R500c, bins=lbins, weights=data.gas.masses.to('Msun'))
        hist = hist / mass_hist
        hist *= unyt.K
        hist = (hist * unyt.boltzmann_constant).to('keV')

    elif weights.lower() == 'entropy':
        weights_field = data.gas.mass_weighted_temperatures * unyt.boltzmann_constant \
                        / data.gas.electron_number_densities ** (2 / 3)
        hist, bin_edges = np.histogram(deltaR / R500c, bins=lbins, weights=weights_field.value)
        mass_hist, _ = np.histogram(deltaR / R500c, bins=lbins, weights=data.gas.masses.to('Msun'))
        hist = hist / mass_hist

    elif weights.lower() == 'pressure':
        weights_field = data.gas.densities * data.gas.mass_weighted_temperatures * unyt.boltzmann_constant / 0.59 \
                        / unyt.mass_hydrogen
        hist, bin_edges = np.histogram(deltaR / R500c, bins=lbins, weights=weights_field.value)
        mass_hist, _ = np.histogram(deltaR / R500c, bins=lbins, weights=data.gas.masses.to('Msun'))
        hist = hist / mass_hist

    else:
        raise ValueError(f"Unrecognized weighting field: {weights}.")

    bin_centre = np.sqrt(bin_edges[1:] * bin_edges[:-1])

    return bin_centre, hist


def _process_single_halo(zoom: Zoom):
    return profile_3d_single_halo(zoom.snapshot_file, zoom.catalog_file, weights='entropy')


# The results of the multiprocessing Pool are returned in the same order as inputs
with Pool() as pool:
    print(f"Analysis mapped onto {cpu_count():d} CPUs.")
    results = pool.map(_process_single_halo, iter(zooms_register))

    # Recast output into a Pandas dataframe for further manipulation
    columns = [
        'bin_centre (Mpc)',
        'entropy',
    ]
    results = pd.DataFrame(list(results), columns=columns)
    results.insert(0, 'Run name', pd.Series(name_list, dtype=str))
    print(results)

plt.plot(results['bin_centre (Mpc)'][0], results['entropy'][0])
plt.xscale('log')
plt.yscale('log')
plt.show()
