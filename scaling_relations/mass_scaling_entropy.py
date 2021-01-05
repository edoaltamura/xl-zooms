# Plot scaling relations for EAGLE-XL tests
import sys
import os
import unyt
import numpy as np
from typing import Tuple
from multiprocessing import Pool, cpu_count
import h5py as h5
import swiftsimio as sw
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Make the register backend visible to the script
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            'zooms'
        )
    )
)

from register import (
    SILENT_PROGRESSBAR,
    zooms_register,
    Zoom,
    Tcut_halogas,
    name_list,
    vr_numbers,
    get_allpaths_from_last,
    get_snip_handles,
    dump_memory_usage,
)

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

fbary = 0.15741  # Cosmic baryon fraction
mean_molecular_weight = 0.59
mean_atomic_weight_per_free_electron = 1.14

entropy_scaling = 'k500'
entropy_radius_r500c = 0.1


def process_single_halo(
        path_to_snap: str,
        path_to_catalogue: str
) -> tuple:
    # Read in halo properties
    with h5.File(f'{path_to_catalogue}', 'r') as h5file:
        XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc)
        YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc)
        ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc)
        M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
        R500c = unyt.unyt_quantity(h5file['/SO_R_500_rhocrit'][0], unyt.Mpc)

    # print(XPotMin, YPotMin, ZPotMin, M500c, R500c)

    # Read in gas particles
    mask = sw.mask(f'{path_to_snap}', spatial_only=False)
    region = [[XPotMin - R500c, XPotMin + R500c],
              [YPotMin - R500c, YPotMin + R500c],
              [ZPotMin - R500c, ZPotMin + R500c]]
    mask.constrain_spatial(region)
    mask.constrain_mask("gas", "temperatures", Tcut_halogas * mask.units.temperature, 1.e12 * mask.units.temperature)
    data = sw.load(f'{path_to_snap}', mask=mask)
    posGas = data.gas.coordinates
    massGas = data.gas.masses * data.gas.masses
    mass_weighted_tempGas = data.gas.temperatures

    # Select hot gas within sphere
    deltaX = posGas[:, 0] - XPotMin
    deltaY = posGas[:, 1] - YPotMin
    deltaZ = posGas[:, 2] - ZPotMin
    deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2)
    index = np.where(deltaR < R500c)[0]
    Mhot500c = np.sum(massGas[index])
    fhot500c = Mhot500c / M500c

    # Calculate entropy
    sphere_index = np.where(deltaR < entropy_radius_r500c * R500c)[0]
    mass_sphere = np.sum(massGas[sphere_index])
    volume_sphere = (4. * np.pi / 3.) * (entropy_radius_r500c * R500c) ** 3
    density_sphere = mass_sphere / volume_sphere

    kBT_sphere = np.sum(mass_weighted_tempGas[sphere_index])
    print(kBT_sphere)
    kBT_sphere *= unyt.boltzmann_constant
    kBT_sphere /= mass_sphere
    kBT_sphere = kBT_sphere.to('keV')

    if entropy_scaling.lower() == 'k500':

        mean_density_R500c = (3 * M500c * fbary / (4 * np.pi * R500c ** 3)).to(density_sphere.units)
        kBT_500crit = unyt.G * mean_molecular_weight * M500c * unyt.mass_proton / 2 / R500c
        kBT_500crit = kBT_500crit.to(kBT_sphere.units)
        # Note: the ratio of densities is the same as ratio of electron number densities
        entropy = kBT_sphere / kBT_500crit * (mean_density_R500c / density_sphere) ** (2 / 3)

    elif entropy_scaling.lower() == 'physical':

        number_density_gas = density_sphere / (mean_molecular_weight * unyt.mass_proton)
        number_density_gas = number_density_gas.to('1/cm**3')
        entropy = kBT_sphere / number_density_gas ** (2 / 3)
        entropy = entropy.to('keV*cm**2')

    return M500c.to(unyt.Solar_Mass), Mhot500c.to(unyt.Solar_Mass), fhot500c, entropy


def _process_single_halo(zoom: Zoom):
    return process_single_halo(zoom.snapshot_file, zoom.catalog_file)


def m_500_entropy():
    vr_num = 'L0300N0564_VR813_+1res_Isotropic'

    _zooms_register = [zoom for zoom in zooms_register if f"{vr_num}" in zoom.run_name]
    _name_list = [zoom_name for zoom_name in name_list if f"{vr_num}" in zoom_name]

    if len(zooms_register) == 1:
        print("Analysing one object only. Not using multiprocessing features.")
        results = [_process_single_halo(_zooms_register[0])]
    else:
        num_threads = len(_zooms_register) if len(_zooms_register) < cpu_count() else cpu_count()
        # The results of the multiprocessing Pool are returned in the same order as inputs
        print(f"Analysis of {len(_zooms_register):d} zooms mapped onto {num_threads:d} CPUs.")
        with Pool(num_threads) as pool:
            results = pool.map(_process_single_halo, iter(_zooms_register))

    # Recast output into a Pandas dataframe for further manipulation
    columns = [
        'M_500crit (M_Sun)',
        'M_hot (< R_500crit) (M_Sun)',
        'f_hot (< R_500crit)',
        'entropy',
    ]
    results = pd.DataFrame(list(results), columns=columns, dtype=np.float64)
    results.insert(0, 'Run name', pd.Series(name_list, dtype=str))
    print(results.head())
    dump_memory_usage()

    fig, ax = plt.subplots()

    # Display zoom data
    for i in range(len(results)):

        marker = ''
        if '-8res' in results.loc[i, "Run name"]:
            marker = '.'
        elif '+1res' in results.loc[i, "Run name"]:
            marker = '^'

        color = ''
        if 'Ref' in results.loc[i, "Run name"]:
            color = 'black'
        elif 'MinimumDistance' in results.loc[i, "Run name"]:
            color = 'orange'
        elif 'Isotropic' in results.loc[i, "Run name"]:
            color = 'lime'

        markersize = 14
        if marker == '.':
            markersize *= 1.5

        ax.scatter(
            results.loc[i, "M_500crit (M_Sun)"],
            results.loc[i, "entropy"],
            marker=marker, c=color, alpha=0.5, s=markersize, edgecolors='none', zorder=5
        )

    # Build legends
    handles = [
        Line2D([], [], marker='.', markeredgecolor='black', markerfacecolor='none', markeredgewidth=1,
               linestyle='None', markersize=6, label='-8 Res'),
        Line2D([], [], marker='^', markeredgecolor='black', markerfacecolor='none', markeredgewidth=1,
               linestyle='None', markersize=3, label='+1 Res'),
        Patch(facecolor='black', edgecolor='None', label='Random (Ref)'),
        Patch(facecolor='orange', edgecolor='None', label='Minimum distance'),
        Patch(facecolor='lime', edgecolor='None', label='Isotropic'),
    ]
    legend_sims = plt.legend(handles=handles, loc=2)

    ax.add_artist(legend_sims)
    ax.set_xlabel(r'$M_{500{\rm crit}}\ [{\rm M}_{\odot}]$')
    ax.set_ylabel(r'Entropy $\ (r<{{0:.1g}} R_{500{\rm crit}})$'.format(entropy_radius_r500c))
    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.savefig(f'{zooms_register[0].output_directory}/m500_k500.png', dpi=300)
    plt.show()
    plt.close()

print(_process_single_halo(zooms_register[0]))