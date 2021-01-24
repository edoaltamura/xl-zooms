# Plot scaling relations for EAGLE-XL tests
import sys
import unyt
import numpy as np
from typing import Tuple
from multiprocessing import Pool, cpu_count
import swiftsimio as sw
import h5py as h5
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Make the register backend visible to the script
sys.path.append("../zooms")
sys.path.append("../observational_data")

from register import zooms_register, Zoom, Tcut_halogas, name_list
import observational_data as obs

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

cosmology = obs.Observations().cosmo_model
fbary = cosmology.Ob0 / cosmology.Om0  # Cosmic baryon fraction


def process_single_halo(
        path_to_snap: str,
        path_to_catalogue: str
) -> Tuple[unyt.unyt_quantity]:
    # Read in halo properties
    with h5.File(f'{path_to_catalogue}', 'r') as h5file:
        XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc)
        YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc)
        ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc)
        R500c = unyt.unyt_quantity(h5file['/SO_R_500_rhocrit'][0], unyt.Mpc)
        R200c = unyt.unyt_quantity(h5file['/R_200crit'][0], unyt.Mpc)
        M500c = unyt.unyt_quantity(
            h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass
        )
        Mbh_aperture50kpc = unyt.unyt_quantity(
            h5file['Aperture_SubgridMasses_aperture_total_bh_50_kpc'][0] * 1.e10, unyt.Solar_Mass
        )
        Mbh_max = unyt.unyt_quantity(
            h5file['/SubgridMasses_max_bh'][0] * 1.e10, unyt.Solar_Mass
        )
        Mstar_bcg_50kpc = unyt.unyt_quantity(
            h5file['/Aperture_mass_star_50_kpc'][0] * 1.e10, unyt.Solar_Mass
        )

        # Read in particles
        mask = sw.mask(f'{path_to_snap}', spatial_only=True)
        region = [[XPotMin - R200c, XPotMin + R200c],
                  [YPotMin - R200c, YPotMin + R200c],
                  [ZPotMin - R200c, ZPotMin + R200c]]
        mask.constrain_spatial(region)
        data = sw.load(f'{path_to_snap}', mask=mask)

        # Get positions for all BHs in the bounding region
        bh_positions = data.black_holes.coordinates
        bh_coordX = bh_positions[:, 0] - XPotMin
        bh_coordY = bh_positions[:, 1] - YPotMin
        bh_coordZ = bh_positions[:, 2] - ZPotMin
        bh_radial_distance = np.sqrt(bh_coordX ** 2 + bh_coordY ** 2 + bh_coordZ ** 2)

        # The central SMBH will probably be massive: filter above 1e8 Msun
        bh_masses = data.black_holes.subgrid_masses.to_physical()
        bh_top_massive_index = np.where(bh_masses.to('Msun').value > 1.e8)[0]

        massive_bh_radial_distances = bh_radial_distance[bh_top_massive_index]
        massive_bh_masses = bh_masses[bh_top_massive_index]

        # Get the central BH closest to centre of halo at z=0
        central_bh_index = np.argmin(bh_radial_distance[bh_top_massive_index])
        central_bh_dr = massive_bh_radial_distances[central_bh_index]
        central_bh_mass = massive_bh_masses[central_bh_index]

    return M500c, Mbh_aperture50kpc, Mbh_max, central_bh_mass, central_bh_dr, Mstar_bcg_50kpc


def _process_single_halo(zoom: Zoom):
    return process_single_halo(zoom.snapshot_file, zoom.catalog_file)


def process_catalogue(find_keyword: str = '', savefile: bool = False) -> pd.DataFrame:
    if find_keyword == '':
        _zooms_register = zooms_register
    else:
        _zooms_register = [zoom for zoom in zooms_register if f"{find_keyword}" in zoom.run_name]

    _name_list = [zoom.run_name for zoom in _zooms_register]

    if len(_zooms_register) == 1:
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
        'M_500crit',
        'Mbh_aperture50kpc',
        'Mbh_max',
        'central_bh_mass',
        'central_bh_dr',
        'Mstar_bcg_50kpc',
    ]
    results = pd.DataFrame(list(results), columns=columns)
    results.insert(0, 'Run name', pd.Series(_name_list, dtype=str))
    print(results.head())

    if savefile:
        file_name = f'{zooms_register[0].output_directory}/median_radial_profiles_catalogue.pkl'
        results.to_pickle(file_name)
        print(f"Catalogue file saved to {file_name}")

    return results


def plot_bhmass_halomass(object_database: pd.DataFrame) -> None:
    fig, ax = plt.subplots()

    for i in range(len(object_database)):

        marker = ''
        if '-8res' in object_database.loc[i, "Run name"]:
            marker = '.'
        elif '+1res' in object_database.loc[i, "Run name"]:
            marker = '^'

        color = ''
        if 'Ref' in object_database.loc[i, "Run name"]:
            color = 'black'
        elif 'MinimumDistance' in object_database.loc[i, "Run name"]:
            color = 'orange'
        elif 'Isotropic' in object_database.loc[i, "Run name"]:
            color = 'lime'

        markersize = 14
        if marker == '.':
            markersize *= 1.5

        ax.scatter(
            object_database.loc[i, "M_500crit"],
            object_database.loc[i, "central_bh_mass"],
            marker=marker, c=color, alpha=0.5, s=markersize, edgecolors='none', zorder=5
        )


    # Display observational data
    Budzynski14 = obs.Budzynski14()
    Kravtsov18 = obs.Kravtsov18()
    ax.plot(Budzynski14.M500_trials, Budzynski14.Mstar_trials_median, linestyle='-', color=(0.8, 0.8, 0.8), zorder=0)
    ax.fill_between(Budzynski14.M500_trials, Budzynski14.Mstar_trials_lower, Budzynski14.Mstar_trials_upper,
                    linewidth=0, color=(0.85, 0.85, 0.85), edgecolor='none', zorder=0)
    ax.scatter(Kravtsov18.M500, Kravtsov18.Mstar500, marker='*', s=12, color=(0.85, 0.85, 0.85),
               linewidth=1.2, edgecolors='none', zorder=0)

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