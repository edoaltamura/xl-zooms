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

    if savefile:
        file_name = f'{zooms_register[0].output_directory}/median_radial_profiles_catalogue.pkl'
        results.to_pickle(file_name)
        print(f"Catalogue file saved to {file_name}")

    return results


def load_catalogue(find_keyword: str = '', filename: str = None) -> pd.DataFrame:

    if filename is None:
        file_path = f'{zooms_register[0].output_directory}/median_radial_profiles_catalogue.pkl'
    else:
        file_path = filename

    print(f"Retrieving catalogue file {file_path}")
    catalogue = pd.read_pickle(file_path)

    if find_keyword != '':
        match_filter = catalogue['Run name'].str.contains(r'{0}'.format(find_keyword), na=False)
        catalogue = catalogue[match_filter]

    print(f"Loaded {len(catalogue):d} objects.")
    return catalogue


def attach_mass_bin_index(object_database: pd.DataFrame, n_bins: int = 3) -> Tuple[pd.DataFrame, np.ndarray]:

    m500crit_log10 = np.array([np.log10(m.value) for m in object_database['M_500crit']])
    bin_log_edges = np.linspace(m500crit_log10.min(), m500crit_log10.max() * 1.01, n_bins + 1)
    bin_indices = np.digitize(m500crit_log10, bin_log_edges)
    object_database.insert(1, 'M_500crit bin_indices', pd.Series(bin_indices, dtype=int))

    return object_database, bin_log_edges


def plot_radial_profiles_median(object_database: pd.DataFrame, bin_edges: np.ndarray) -> None:
    from matplotlib.cm import get_cmap

    name = "Set2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors = cmap.colors  # type: list

    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=colors)

    # Display zoom data
    for i, bin_edge in enumerate(bin_edges[:-1]):
        bin_select = object_database['M_500crit bin_indices'] == i + 1
        plot_database = object_database[bin_select]
        max_convergence_radius = plot_database['convergence_radius'].max()

        # Plot only profiles outside the *largest* convergence radius
        radial_profiles = []
        for j in range(len(plot_database)):
            convergence_index = np.where(plot_database.iloc[j]['bin_centre'] > max_convergence_radius)[0]
            radial_profiles.append(plot_database.iloc[j][FIELD_NAME][convergence_index])

        radial_profiles = np.asarray(radial_profiles)
        convergence_index = np.where(plot_database.iloc[0]['bin_centre'] > max_convergence_radius)[0]
        bin_centres = plot_database.iloc[0]['bin_centre'][convergence_index]
        median_profile = np.median(radial_profiles, axis=0)
        percent16_profile = np.percentile(radial_profiles, 16, axis=0)
        percent84_profile = np.percentile(radial_profiles, 84, axis=0)

        ax.fill_between(
            bin_centres, percent84_profile, percent16_profile,
            linewidth=0, alpha=0.5, color=colors[i],
        )
        ax.plot(
            bin_centres, median_profile,
            linestyle='-', linewidth=1, alpha=1, color=colors[i],
            label=f"$10^{{{bin_edges[i]:.1f}}}<M_{{500, crit}}/M_{{\odot}}<10^{{{bin_edges[i + 1]:.1f}}}$"
        )

    ax.set_xlabel(r'$R/R_{500{\rm crit}}$')
    ax.set_ylabel(plot_database.iloc[0]['ylabel'])

    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.title(keyword)
    fig.savefig(f'{zooms_register[0].output_directory}/median_radial_profiles_{keyword}.png', dpi=300)
    plt.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":

    dts = ['7.5', '8', '8.5', '9', '9.5']

    for dt in dts:
        keyword = f'fixedAGNdT{dt}_'

        try:
            results_database = load_catalogue(find_keyword=keyword)
        except FileNotFoundError or FileExistsError as err:
            print(err, "\nAnalysing catalogues from data...")
            results_database = process_catalogue(find_keyword=keyword, savefile=False)

        results_database, bin_edges = attach_mass_bin_index(results_database)
        plot_radial_profiles_median(results_database, bin_edges)
