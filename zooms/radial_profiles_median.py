import sys
import os
import unyt
from typing import Tuple
import numpy as np
from multiprocessing import Pool, cpu_count
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
            'scaling_relations'
        )
    )
)
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            'zooms'
        )
    )
)

from radial_profiles import profile_3d_single_halo as profiles
from mass_scaling_entropy import process_single_halo as entropy_scaling
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

FIELD_NAME = 'entropy_physical'


def _process_single_halo(zoom: Zoom) -> tuple:
    scaling_database = entropy_scaling(zoom.snapshot_file, zoom.catalog_file)
    profiles_database = profiles(zoom.snapshot_file, zoom.catalog_file, weights=FIELD_NAME)
    return tuple(scaling_database + profiles_database)


def process_catalogue() -> pd.DataFrame:

    find_keyword = '-8res_MinimumDistance_fixedAGNdT8_Nheat1_SNnobirth'

    if find_keyword == '':
        _zooms_register = zooms_register
    else:
        _zooms_register = [zoom for zoom in zooms_register if find_keyword in zoom.run_name]

    _name_list = [zoom.run_name for zoom in _zooms_register]
    for i in _name_list: print(i)

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
        'M_hot (< R_500crit)',
        'f_hot (< R_500crit)',
        'entropy',
        'kBT_500crit',
        'bin_centre',
        FIELD_NAME,
        'ylabel',
        'convergence_radius'
    ]
    results = pd.DataFrame(list(results), columns=columns)
    results.insert(0, 'Run name', pd.Series(_name_list, dtype=str))
    print(results.head())
    dump_memory_usage()
    return results


def attach_mass_bin_index(object_database: pd.DataFrame, n_bins: int = 3) -> Tuple[pd.DataFrame, np.ndarray]:
    bin_log_edges = np.logspace(np.min(object_database['M_500crit']), np.max(object_database['M_500crit']), n_bins)
    bin_indices = np.digitize(object_database['M_500crit'], bin_log_edges)
    object_database.insert(1, 'M_500crit bin_indices', pd.Series(bin_indices, dtype=int))
    return object_database, bin_log_edges


def plot_radial_profiles_median(object_database: pd.DataFrame, bin_edges: np.ndarray) -> None:
    from matplotlib.cm import get_cmap

    name = "Set2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors = cmap.colors   # type: list

    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=colors)

    # Display zoom data
    for i, bin_edge in enumerate(bin_edges[:-1]):
        bin_select = object_database['M_500crit bin_indices'] == i
        plot_database = object_database[bin_select]
        max_convergence_radius = plot_database['convergence_radius'].max()

        # Plot only profiles outside the *largest* convergence radius
        profile_size = len(np.where(plot_database['bin_centre'][0] > max_convergence_radius)[0])
        radial_profiles = np.zeros((len(plot_database), profile_size))
        for j in range(len(plot_database)):
            convergence_index = np.where(plot_database['bin_centre'][j] > max_convergence_radius)[0]
            radial_profiles[i] = plot_database[FIELD_NAME][j][convergence_index]

        bin_centres = plot_database['bin_centre'][0][convergence_index]
        median_profile = np.median(radial_profiles, axis=0)

        ax.plot(
            bin_centres, median_profile,
            linestyle='-', linewidth=0.5, alpha=1,
            label=f"{bin_edges[i]}$<M_{{500, crit}}<${bin_edges[i + 1]}"
        )

    ax.set_xlabel(r'$R/R_{500{\rm crit}}$')
    ax.set_ylabel(plot_database['ylabel'][0])

    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.savefig(f'{zooms_register[0].output_directory}/median_radial_profiles.png', dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    results_database = process_catalogue()
    results_database, bin_edges = attach_mass_bin_index(results_database)
    plot_radial_profiles_median(results_database, bin_edges)
