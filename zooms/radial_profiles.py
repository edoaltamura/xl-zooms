import unyt
import numpy as np
from typing import Tuple, Union
from multiprocessing import Pool, cpu_count
import h5py as h5
from copy import deepcopy
import swiftsimio as sw
import pandas as pd
import matplotlib.pyplot as plt

from register import zooms_register, Zoom, Tcut_halogas, name_list

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

# Constants
bins = 30
radius_bounds = [0.05, 4]  # In units of R500crit
fbary = 0.15741  # Cosmic baryon fraction
mean_molecular_weight = 0.59
mean_atomic_weight_per_free_electron = 1.14


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


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

    # Since useful for different applications, attach datasets
    data.gas.mass_weighted_temperatures = data.gas.masses * data.gas.temperatures
    data.gas.masses = data.gas.masses

    # Rescale profiles to R500c
    radial_distance = deltaR / R500c
    assert radial_distance.units == unyt.dimensionless

    # Construct bins for mass-weighted quantities and retrieve bin_edges
    lbins = np.logspace(np.log10(radius_bounds[0]), np.log10(radius_bounds[1]), bins) * radial_distance.units
    mass_weights, bin_edges = histogram_unyt(radial_distance, bins=lbins, weights=data.gas.masses)
    bin_centre = np.sqrt(bin_edges[1:] * bin_edges[:-1])

    # Allocate weights
    if weights.lower() == 'gas_mass':
        hist = mass_weights / M500c.to(mass_weights.units)

        ylabel = r'$M(dR) / M_{500{\rm crit}}$'

    elif weights.lower() == 'gas_mass_cumulative':
        hist = cumsum_unyt(mass_weights) / M500c.to(mass_weights.units)

        ylabel = r'$M(<R) / M_{500{\rm crit}}$'

    elif weights.lower() == 'gas_density':
        hist, _ = histogram_unyt(radial_distance, bins=lbins, weights=data.gas.densities)
        hist /= rho_crit.to(hist.units)
        hist *= bin_centre ** 2

        ylabel = r'$(\rho_{\rm gas}/\rho_{\rm crit})\ (R/R_{500{\rm crit}})^3 $'

    elif weights.lower() == 'mass_weighted_temps':
        weights_field = data.gas.mass_weighted_temperatures
        hist, _ = np.histogram(radial_distance, bins=lbins, weights=weights_field)
        hist /= mass_weights

        ylabel = r'$T$ [K]'

    elif weights.lower() == 'mass_weighted_temps_kev':
        weights_field = data.gas.mass_weighted_temperatures
        hist, _ = histogram_unyt(radial_distance, bins=lbins, weights=weights_field)
        hist /= mass_weights
        hist = (hist * unyt.boltzmann_constant).to('keV')

        # Make dimensionless, divide by (k_B T_500crit)
        norm = unyt.G * mean_molecular_weight * M500c * unyt.mass_proton / 2 / R500c
        norm = norm.to('keV')
        hist /= norm

        ylabel = r'$(k_B T/k_B T_{500{\rm crit}})$'

    elif weights.lower() == 'entropy':
        weights_field = data.gas.entropies
        hist, _ = histogram_unyt(radial_distance, bins=lbins, weights=weights_field)

        # Make dimensionless, divide by K_500crit
        norm = unyt.G * mean_molecular_weight * M500c * unyt.mass_proton / 2 / R500c \
               / (500 * fbary * rho_crit / mean_atomic_weight_per_free_electron / unyt.mass_proton) ** (2 / 3)

        hist /= norm.to(hist.units)

        ylabel = r'$(K/K_{500{\rm crit}})$'

    elif weights.lower() == 'pressure':
        weights_field = data.gas.pressures
        hist, _ = histogram_unyt(radial_distance, bins=lbins, weights=weights_field)

        # Make dimensionless, divide by P_500crit
        norm = 500 * fbary * rho_crit * unyt.G * M500c / 2 / R500c
        hist /= norm.to(hist.units)
        hist *= bin_centre ** 3

        ylabel = r'$(P/P_{500{\rm crit}})\ (R/R_{500{\rm crit}})^3 $'

    else:
        raise ValueError(f"Unrecognized weighting field: {weights}.")

    return bin_centre, hist, ylabel

field_name = 'pressure'

def _process_single_halo(zoom: Zoom):
    return profile_3d_single_halo(zoom.snapshot_file, zoom.catalog_file, weights=field_name)


# The results of the multiprocessing Pool are returned in the same order as inputs
with Pool() as pool:
    print(f"Analysis mapped onto {cpu_count():d} CPUs.")
    results = pool.map(_process_single_halo, iter(zooms_register))

    # Recast output into a Pandas dataframe for further manipulation
    columns = [
        'bin_centre',
        field_name,
        'ylabel'
    ]
    results = pd.DataFrame(list(results), columns=columns)
    results.insert(0, 'Run name', pd.Series(name_list, dtype=str))
    print(results)

fig, ax = plt.subplots()
for i in range(len(results)):

    style = ''
    if '-8res' in results.loc[i, "Run name"]:
        style = '-'
    elif '+1res' in results.loc[i, "Run name"]:
        style = ':'

    color = ''
    if 'Ref' in results.loc[i, "Run name"]:
        color = 'black'
    elif 'MinimumDistance' in results.loc[i, "Run name"]:
        color = 'orange'
    elif 'Isotropic' in results.loc[i, "Run name"]:
        color = 'lime'

    ax.plot(results['bin_centre'][i], results[field_name][i], linestyle=style, linewidth=0.3, color=color, alpha=0.4)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$R/R_{500{\rm crit}}$')
ax.set_ylabel(results['ylabel'][0])
plt.show()
