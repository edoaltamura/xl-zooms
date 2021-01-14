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

from register import zooms_register, Zoom, Tcut_halogas, name_list
import observational_data as obs

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

fbary = 0.15741  # Cosmic baryon fraction


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
    massGas = data.gas.masses

    # Select hot gas within sphere
    deltaX = posGas[:, 0] - XPotMin
    deltaY = posGas[:, 1] - YPotMin
    deltaZ = posGas[:, 2] - ZPotMin
    deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2)
    index = np.where(deltaR < R500c)[0]
    Mhot500c = np.sum(massGas[index])
    fhot500c = Mhot500c / M500c

    return M500c.to(unyt.Solar_Mass), Mhot500c.to(unyt.Solar_Mass), fhot500c.value


def _process_single_halo(zoom: Zoom):
    return process_single_halo(zoom.snapshot_file, zoom.catalog_file)


def m_500_hotgas():
    vr_num = 'fixedAGNdT'

    _zooms_register = [zoom for zoom in zooms_register if f"{vr_num}" in zoom.run_name]
    _name_list = [zoom.run_name for zoom in _zooms_register]

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
    ]
    results = pd.DataFrame(list(results), columns=columns, dtype=np.float64)
    results.insert(0, 'Run name', pd.Series(name_list, dtype=str))
    print(results.head())

    # Display zoom data
    fig, ax = plt.subplots()
    for i in range(len(results)):

        marker = ''
        if '-8res' in results.loc[i, "Run name"]:
            marker = '.'
        elif '+1res' in results.loc[i, "Run name"]:
            marker = '^'

            # color = ''
            # if 'Ref' in results.loc[i, "Run name"]:
            #     color = '#660099'
            # elif 'MinimumDistance' in results.loc[i, "Run name"]:
            #     color = 'orange'
            # elif 'Isotropic' in results.loc[i, "Run name"]:
            #     color = 'lime'

            color = ''
            if 'dT9.5_' in results.loc[i, "Run name"]:
                color = 'blue'
            elif 'dT9_' in results.loc[i, "Run name"]:
                color = 'black'
            elif 'dT8.5_' in results.loc[i, "Run name"]:
                color = 'red'
            elif 'dT8_' in results.loc[i, "Run name"]:
                color = 'orange'
            elif 'dT7.5_' in results.loc[i, "Run name"]:
                color = 'lime'

        markersize = 14
        if marker == '.':
            markersize *= 1.5

        ax.scatter(
            results.loc[i, "M_500crit (M_Sun)"],
            results.loc[i, "M_hot (< R_500crit) (M_Sun)"],
            marker=marker, c=color, alpha=0.7, s=markersize, edgecolors='none', zorder=5
        )

        if 'Ref' in results.loc[i, "Run name"]:
            print(results.loc[i, "M_500crit (M_Sun)"] / 1.e10, results.loc[i, "M_hot (< R_500crit) (M_Sun)"] / 1.e10)

    # Display observational data
    Sun09 = obs.Sun09()
    Lovisari15 = obs.Lovisari15()
    ax.scatter(Sun09.M500, Sun09.Mgas500, marker='d', s=8, alpha=1,
               color=(0.65, 0.65, 0.65), edgecolors='none', zorder=0)
    ax.scatter(Lovisari15.M500, Lovisari15.Mgas500, marker='s', s=8, alpha=1,
               color=(0.65, 0.65, 0.65), edgecolors='none', zorder=0)

    # Build legends
    handles = [
        Line2D([], [], marker='.', markeredgecolor='black', markerfacecolor='none', markeredgewidth=1,
               linestyle='None', markersize=6, label='-8 Res'),
        Line2D([], [], marker='^', markeredgecolor='black', markerfacecolor='none', markeredgewidth=1,
               linestyle='None', markersize=3, label='+1 Res'),
        # Patch(facecolor='black', edgecolor='None', label='Random (Ref)'),
        # Patch(facecolor='orange', edgecolor='None', label='Minimum distance'),
        # Patch(facecolor='lime', edgecolor='None', label='Isotropic'),
        Patch(facecolor='blue', edgecolor='None', label='dT9.5'),
        Patch(facecolor='black', edgecolor='None', label='dT9'),
        Patch(facecolor='red', edgecolor='None', label='dT8.5'),
        Patch(facecolor='orange', edgecolor='None', label='dT8'),
        Patch(facecolor='lime', edgecolor='None', label='dT7.5'),
    ]
    legend_sims = plt.legend(handles=handles, loc=2)
    handles = [
        Line2D([], [], color=(0.65, 0.65, 0.65), marker='d', markeredgecolor='none', linestyle='None', markersize=4,
               label=Sun09.paper_name),
        Line2D([], [], color=(0.65, 0.65, 0.65), marker='s', markeredgecolor='none', linestyle='None', markersize=4,
               label=Lovisari15.paper_name),
        Line2D([], [], color='black', linestyle='--', markersize=0, label=f"Planck18 $f_{{bary}}=${fbary:.3f}"),
    ]
    del Sun09, Lovisari15
    legend_obs = plt.legend(handles=handles, loc=4)
    ax.add_artist(legend_sims)
    ax.add_artist(legend_obs)

    ax.set_xlabel(r'$M_{500{\rm crit}}\ [{\rm M}_{\odot}]$')
    ax.set_ylabel(r'$M_{{\rm gas},500{\rm crit}}\ [{\rm M}_{\odot}]$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(ax.get_xlim(), [lim * fbary for lim in ax.get_xlim()], '--', color='k')

    fig.savefig(f'{zooms_register[0].output_directory}/m_500_hotgas.png', dpi=300)
    plt.show()
    plt.close()

    return


def f_500_hotgas():
    fig, ax = plt.subplots()

    # The results of the multiprocessing Pool are returned in the same order as inputs
    with Pool() as pool:
        print(f"Analysis mapped onto {cpu_count():d} CPUs.")
        results = pool.map(_process_single_halo, iter(zooms_register))

        # Recast output into a Pandas dataframe for further manipulation
        columns = [
            'M_500crit (M_Sun)',
            'M_hot (< R_500crit) (M_Sun)',
            'f_hot (< R_500crit)',
        ]
        results = pd.DataFrame(list(results), columns=columns, dtype=np.float64)
        results.insert(0, 'Run name', pd.Series(name_list, dtype=str))
        print(results.head())

    # Display zoom data
    for i in range(len(results)):

        marker = ''
        if '-8res' in results.loc[i, "Run name"]:
            marker = '.'
        elif '+1res' in results.loc[i, "Run name"]:
            marker = '^'

        # color = ''
        # if 'Ref' in results.loc[i, "Run name"]:
        #     color = '#660099'
        # elif 'MinimumDistance' in results.loc[i, "Run name"]:
        #     color = 'orange'
        # elif 'Isotropic' in results.loc[i, "Run name"]:
        #     color = 'lime'

        color = ''
        if 'dT9.5_' in results.loc[i, "Run name"]:
            color = 'blue'
        elif 'dT9_' in results.loc[i, "Run name"]:
            color = 'black'
        elif 'dT8.5_' in results.loc[i, "Run name"]:
            color = 'red'
        elif 'dT8_' in results.loc[i, "Run name"]:
            color = 'orange'
        elif 'dT7.5_' in results.loc[i, "Run name"]:
            color = 'lime'

        markersize = 14
        if marker == '.':
            markersize *= 1.5

        ax.scatter(
            results.loc[i, "M_500crit (M_Sun)"],
            results.loc[i, "f_hot (< R_500crit)"],
            marker=marker, c=color, alpha=0.5, s=markersize, edgecolors='none', zorder=5
        )

    # Display observational data
    Sun09 = obs.Sun09()
    Lovisari15 = obs.Lovisari15()
    ax.scatter(Sun09.M500, Sun09.Mgas500 / Sun09.M500, marker='d', s=8, alpha=1,
               color=(0.95, 0.95, 0.95), edgecolors='none', zorder=0)
    ax.scatter(Lovisari15.M500, Lovisari15.Mgas500 / Lovisari15.M500, marker='s', s=8, alpha=1,
               color=(0.95, 0.95, 0.95), edgecolors='none', zorder=0)

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
        Line2D([], [], color=(0.95, 0.95, 0.95), marker='d', markeredgecolor='none', linestyle='None', markersize=4,
               label=Sun09.paper_name),
        Line2D([], [], color=(0.95, 0.95, 0.95), marker='s', markeredgecolor='none', linestyle='None', markersize=4,
               label=Lovisari15.paper_name),
        Line2D([], [], color='black', linestyle='--', markersize=0, label=f"Planck18 $f_{{bary}}=${fbary:.3f}"),
    ]
    del Sun09, Lovisari15
    legend_obs = plt.legend(handles=handles, loc=4)
    # ax.add_artist(legend_sims)
    ax.add_artist(legend_obs)

    ax.set_xlabel(r'$M_{500{\rm crit}}\ [{\rm M}_{\odot}]$')
    ax.set_ylabel(r'$f_{{\rm gas},500{\rm crit}}$')
    ax.set_xscale('log')
    ax.plot(ax.get_xlim(), [fbary for lim in ax.get_xlim()], '--', color='k')

    fig.savefig(f'{zooms_register[0].output_directory}/f_500_hotgas.png', dpi=300)
    plt.show()
    plt.close()

    return


m_500_hotgas()
# f_500_hotgas()
