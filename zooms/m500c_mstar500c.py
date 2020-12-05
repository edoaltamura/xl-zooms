# Plot scaling relations for EAGLE-XL tests
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

    # Select hot gas within sphere
    deltaX = posGas[:, 0] - XPotMin
    deltaY = posGas[:, 1] - YPotMin
    deltaZ = posGas[:, 2] - ZPotMin
    deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2)
    index = np.where(deltaR < R500c)[0]
    Mhot500c = np.sum(massGas[index])

    return M500c.to(unyt.Solar_Mass), Mstar500c.to(unyt.Solar_Mass), Mhot500c.to(unyt.Solar_Mass)


def _process_single_halo(zoom: Zoom):
    return process_single_halo(zoom.snapshot_file, zoom.catalog_file)


def m_500_star():
    fig, ax = plt.subplots()

    # The results of the multiprocessing Pool are returned in the same order as inputs
    with Pool() as pool:
        print(f"Analysis mapped onto {cpu_count():d} CPUs.")
        results = pool.map(_process_single_halo, iter(zooms_register))

        # Recast output into a Pandas dataframe for further manipulation
        columns = [
            'M_500crit (M_Sun)',
            'M_star (< R_500crit) (M_Sun)',
            'M_hot (< R_500crit) (M_Sun)',
        ]
        results = pd.DataFrame(list(results), columns=columns)
        results.insert(0, 'Run name', pd.Series(name_list, dtype=str))
        print(results)

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
            results.loc[i, "M_star (< R_500crit) (M_Sun)"],
            marker=marker, c=color, alpha=0.5, s=markersize, edgecolors='none', zorder=5
        )

    # Display observational data
    Budzynski14 = obs.Budzynski14()
    Kravtsov18 = obs.Kravtsov18()
    ax.plot(Budzynski14.M500_trials, Budzynski14.Mstar_trials_median, linestyle='-', color=(0.8, 0.8, 0.8), zorder=0)
    ax.fill_between(Budzynski14.M500_trials, Budzynski14.Mstar_trials_lower, Budzynski14.Mstar_trials_upper,
                    linestyle='-', color=(0.85, 0.85, 0.85), edgecolor='none', zorder=0)
    ax.scatter(Kravtsov18.M500, Kravtsov18.Mstar500, marker='*', s=12, color=(0.85, 0.85, 0.85),
               edgecolors='none', zorder=0)

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
        Line2D([], [], color=(0.85, 0.85, 0.85), marker='.', markeredgecolor='none', linestyle='-', markersize=0,
               label=Budzynski14.paper_name),
        Line2D([], [], color=(0.85, 0.85, 0.85), marker='*', markeredgecolor='none', linestyle='None', markersize=4,
               label=Kravtsov18.paper_name),
    ]
    del Budzynski14, Kravtsov18
    legend_obs = plt.legend(handles=handles, loc=4)
    ax.add_artist(legend_sims)
    ax.add_artist(legend_obs)

    ax.set_xlabel(r'$M_{500{\rm crit}}\ [{\rm M}_{\odot}]$')
    ax.set_ylabel(r'$M_{{\rm star},500{\rm crit}}\ [{\rm M}_{\odot}]$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{zooms_register[0].output_directory}/m_500_star.png', dpi=300)
    plt.show()
    plt.close()

    return


def f_500_star():
    fig, ax = plt.subplots()

    # The results of the multiprocessing Pool are returned in the same order as inputs
    with Pool() as pool:
        print(f"Analysis mapped onto {cpu_count():d} CPUs.")
        results = pool.map(_process_single_halo, iter(zooms_register))

        # Recast output into a Pandas dataframe for further manipulation
        columns = [
            'M_500crit (M_Sun)',
            'M_star (< R_500crit) (M_Sun)',
            'M_hot (< R_500crit) (M_Sun)',
        ]
        results = pd.DataFrame(list(results), columns=columns)
        results.insert(0, 'Run name', pd.Series(name_list, dtype=str))
        print(results)

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
            results.loc[i, "M_star (< R_500crit) (M_Sun)"] / results.loc[i, "M_500crit (M_Sun)"],
            marker=marker, c=color, alpha=0.5, s=markersize, edgecolors='none', zorder=5
        )

    # Display observational data
    Budzynski14 = obs.Budzynski14()
    Kravtsov18 = obs.Kravtsov18()
    ax.plot(Budzynski14.M500_trials, Budzynski14.Mstar_trials_median / Budzynski14.M500_trials,
            linestyle='-', color=(0.8, 0.8, 0.8), zorder=0)
    ax.fill_between(Budzynski14.M500_trials,
                    Budzynski14.Mstar_trials_lower / Budzynski14.M500_trials,
                    Budzynski14.Mstar_trials_upper / Budzynski14.M500_trials,
                    linestyle='-', color=(0.85, 0.85, 0.85), edgecolor='none', zorder=0)
    ax.scatter(Kravtsov18.M500, Kravtsov18.Mstar500 / Kravtsov18.M500, marker='*', s=12, color=(0.85, 0.85, 0.85),
               edgecolors='none', zorder=0)

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
        Line2D([], [], color=(0.85, 0.85, 0.85), marker='.', markeredgecolor='none', linestyle='-', markersize=0,
               label=Budzynski14.paper_name),
        Line2D([], [], color=(0.85, 0.85, 0.85), marker='*', markeredgecolor='none', linestyle='None', markersize=4,
               label=Kravtsov18.paper_name),
        Line2D([], [], color='black', linestyle='--', markersize=0, label=f"Planck18 $f_{{bary}}=${fbary:.3f}"),
    ]
    del Budzynski14, Kravtsov18
    legend_obs = plt.legend(handles=handles, loc=4)
    ax.add_artist(legend_sims)
    ax.add_artist(legend_obs)
    ax.plot(ax.get_xlim(), [fbary for lim in ax.get_xlim()], '--', color='k')

    ax.set_xlabel(r'$M_{500{\rm crit}}\ [{\rm M}_{\odot}]$')
    ax.set_ylabel(r'$f_{{\rm star},500{\rm crit}}$')
    ax.set_xscale('log')
    fig.savefig(f'{zooms_register[0].output_directory}/f_500_star.png', dpi=300)
    plt.show()
    plt.close()

    return


def f_500_bary():
    fig, ax = plt.subplots()

    # The results of the multiprocessing Pool are returned in the same order as inputs
    with Pool() as pool:
        print(f"Analysis mapped onto {cpu_count():d} CPUs.")
        results = pool.map(_process_single_halo, iter(zooms_register))

        # Recast output into a Pandas dataframe for further manipulation
        columns = [
            'M_500crit (M_Sun)',
            'M_star (< R_500crit) (M_Sun)',
            'M_hot (< R_500crit) (M_Sun)',
        ]
        results = pd.DataFrame(list(results), columns=columns)
        results.insert(0, 'Run name', pd.Series(name_list, dtype=str))
        print(results)

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
            (results.loc[i, "M_star (< R_500crit) (M_Sun)"] + results.loc[i, "M_hot (< R_500crit) (M_Sun)"]) \
            / results.loc[i, "M_500crit (M_Sun)"],
            marker=marker, c=color, alpha=0.5, s=markersize, edgecolors='none', zorder=5
        )

    # Display observational data
    Budzynski14 = obs.Budzynski14()
    Kravtsov18 = obs.Kravtsov18()
    # ax.plot(Budzynski14.M500_trials, Budzynski14.Mstar_trials_median / Budzynski14.M500_trials,
    #         linestyle='-', color=(0.8, 0.8, 0.8), zorder=0)
    # ax.fill_between(Budzynski14.M500_trials,
    #                 Budzynski14.Mstar_trials_lower / Budzynski14.M500_trials,
    #                 Budzynski14.Mstar_trials_upper / Budzynski14.M500_trials,
    #                 linestyle='-', color=(0.85, 0.85, 0.85), edgecolor='none', zorder=0)
    # ax.scatter(Kravtsov18.M500, Kravtsov18.Mstar500 / Kravtsov18.M500, marker='*', s=12, color=(0.85, 0.85, 0.85),
    #            edgecolors='none', zorder=0)

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
        Line2D([], [], color=(0.85, 0.85, 0.85), marker='.', markeredgecolor='none', linestyle='-', markersize=0,
               label=Budzynski14.paper_name),
        Line2D([], [], color=(0.85, 0.85, 0.85), marker='*', markeredgecolor='none', linestyle='None', markersize=4,
               label=Kravtsov18.paper_name),
    ]
    del Budzynski14, Kravtsov18
    legend_obs = plt.legend(handles=handles, loc=4)
    ax.add_artist(legend_sims)
    ax.add_artist(legend_obs)

    ax.set_xlabel(r'$M_{500{\rm crit}}\ [{\rm M}_{\odot}]$')
    ax.set_ylabel(r'$f_{{\rm star},500{\rm crit}}$')
    ax.set_xscale('log')
    fig.savefig(f'{zooms_register[0].output_directory}/f_500_bary.png', dpi=300)
    plt.show()
    plt.close()

    return


m_500_star()
f_500_star()
f_500_bary()