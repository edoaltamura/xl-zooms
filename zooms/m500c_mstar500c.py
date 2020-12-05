# Plot scaling relations for EAGLE-XL tests
import os
import unyt
import numpy as np
from typing import Tuple
from multiprocessing import Pool
import h5py as h5
import swiftsimio as sw
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from register import zooms_register, Zoom, Tcut_halogas, name_list
import observational_data as obs

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass


def process_single_halo(
        path_to_snap: str,
        path_to_catalogue: str
) -> Tuple[float]:
    # Read in halo properties
    with h5.File(f'{path_to_catalogue}', 'r') as h5file:
        XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc)
        YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc)
        ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc)
        M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
        Mstar500c = unyt.unyt_quantity(h5file['/SO_Mass_star_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
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
    massGas = data.gas.masses
    tempGas = data.gas.temperatures

    # Select hot gas within sphere
    deltaX = posGas[:, 0] - XPotMin
    deltaY = posGas[:, 1] - YPotMin
    deltaZ = posGas[:, 2] - ZPotMin
    deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2)
    index = np.where(deltaR < R500c)[0]
    Mhot500c = np.sum(massGas[index])

    return M500c, Mstar500c, Mhot500c


def _process_single_halo(zoom: Zoom):
    return process_single_halo(zoom.snapshot_file, zoom.catalog_file)


def make_single_image():
    fig, ax = plt.subplots()

    columns = [
        # 'Run name',
        'M_500crit (* 1e13 M_Sun)',
        'M_star (< R_500crit) (* 1e13 M_Sun)',
        'M_hot (< R_500crit) (* 1e13 M_Sun)',
    ]

    # The results of the multiprocessing Pool are returned in the same order as inputs
    with Pool() as pool:
        results = pool.map(_process_single_halo, iter(zooms_register))
        results = pd.DataFrame(list(results), columns=columns)

    print(results)
    # Display zoom data
    for i, data in enumerate(results):
        M500c = data[0].value
        Mstar500c = data[1].value
        Mhot500c = data[2].value

        print((
            f"{zooms_register[i].run_name:<40s} "
            f"{(M500c / 1.e13):<6.4f} * 1e13 Msun "
            f"{(Mstar500c / 1.e13):<6.4f} * 1e13 Msun "
            f"{(Mhot500c / 1.e13):<6.4f} "
        ))

        marker = ''
        if '-8res' in zooms_register[i].run_name:
            marker = '.'
        elif '+1res' in zooms_register[i].run_name:
            marker = '^'

        color = ''
        if 'Ref' in zooms_register[i].run_name:
            color = 'black'
        elif 'MinimumDistance' in zooms_register[i].run_name:
            color = 'orange'
        elif 'Isotropic' in zooms_register[i].run_name:
            color = 'lime'

        ax.scatter(M500c, Mstar500c, marker=marker, c=color, alpha=0.7, s=15, edgecolors='none')

    # Display observational data
    Budzynski14 = obs.Budzynski14()
    Kravtsov18 = obs.Kravtsov18()
    ax.plot(Budzynski14.M500, Budzynski14.Mstar500, linestyle='-', color='gray')
    ax.scatter(Kravtsov18.M500, Kravtsov18.Mstar500, marker='*', alpha=0.7, s=10, color='gray', edgecolors='none')

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
    handles = [
        Line2D([], [], color='grey', marker='.', markeredgecolor='none', linestyle='-', markersize=0,
               label=Budzynski14.paper_name),
        Line2D([], [], color='grey', marker='*', markeredgecolor='none', linestyle='None', markersize=4,
               label=Kravtsov18.paper_name),
    ]
    legend_obs = plt.legend(handles=handles, loc=4)
    ax.add_artist(legend_sims)
    ax.add_artist(legend_obs)

    ax.set_xlabel(r'$M_{500{\rm crit}}\ [{\rm M}_{\odot}]$')
    ax.set_ylabel(r'$M_{{\rm star},500{\rm crit}}\ [{\rm M}_{\odot}]$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{zooms_register[0].output_directory}/m500c_mstar500c.png', dpi=300)
    plt.show()
    plt.close()

    return


make_single_image()
