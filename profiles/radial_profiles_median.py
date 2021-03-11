import os
import sys
import unyt
import argparse
import h5py as h5
import numpy as np
import pandas as pd
import swiftsimio as sw
from typing import Tuple
import matplotlib.pyplot as plt

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

# Make the register backend visible to the script
sys.path.append("../observational_data")
sys.path.append("../scaling_relations")
sys.path.append("../zooms")

from register import zooms_register, Zoom, Tcut_halogas, calibration_zooms
import observational_data as obs
import scaling_utils as utils
import scaling_style as style
from convergence_radius import convergence_radius
from radial_profiles import profile_3d_single_halo as profiles
from mass_scaling_entropy import process_single_halo as entropy_scaling

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--keywords', type=str, nargs='+', required=True)
parser.add_argument('-e', '--observ-errorbars', default=False, required=False, action='store_true')
parser.add_argument('-r', '--redshift-index', type=int, default=37, required=False,
                    choices=list(range(len(calibration_zooms.get_snap_redshifts()))))
parser.add_argument('-m', '--mass-estimator', type=str.lower, default='crit', required=True,
                    choices=['crit', 'true', 'hse'])
parser.add_argument('-q', '--quiet', default=False, required=False, action='store_true')
args = parser.parse_args()

FIELD_NAME = 'entropy'


@utils.set_scaling_relation_name(os.path.splitext(os.path.basename(__file__))[0])
@utils.set_output_names([
    'M_500crit',
    'M_hot (< R_500crit)',
    'f_hot (< R_500crit)',
    'entropy (= R_500crit)',
    'kBT_500crit',
    'bin_centre',
    FIELD_NAME,
    'ylabel',
    'convergence_radius'
])
def _process_single_halo(zoom: Zoom):
    # Select redshift
    snapshot_file = zoom.get_redshift(args.redshift_index).snapshot_path
    catalog_file = zoom.get_redshift(args.redshift_index).catalogue_properties_path

    if args.mass_estimator == 'crit' or args.mass_estimator == 'true':

        scaling_database = entropy_scaling(snapshot_file, catalog_file)
        profiles_database = profiles(snapshot_file, catalog_file, weights=FIELD_NAME)
        return tuple(scaling_database + profiles_database)

    elif args.mass_estimator == 'hse':
        try:
            hse_catalogue = pd.read_pickle(f'{calibration_zooms.output_directory}/hse_massbias.pkl')
        except FileExistsError as error:
            raise FileExistsError(
                f"{error}\nPlease, consider first generating the HSE catalogue for better performance."
            )

        hse_catalogue_names = hse_catalogue['Run name'].values.tolist()
        print(f"Looking for HSE results in the catalogue - zoom: {zoom.run_name}")
        if zoom.run_name in hse_catalogue_names:
            i = hse_catalogue_names.index(zoom.run_name)
            hse_entry = hse_catalogue.loc[i]
        else:
            raise ValueError(f"{zoom.run_name} not found in HSE catalogue. Please, regenerate the catalogue.")

        scaling_database = entropy_scaling(snapshot_file, catalog_file)
        profiles_database = profiles(snapshot_file, catalog_file, weights=FIELD_NAME, hse_dataset=hse_entry)
        return tuple(scaling_database + profiles_database)


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
    m500crit_log10 = np.array([np.log10(m.value) for m in object_database['M_500crit'].values])
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
    for i in enumerate(bin_edges[:-1]):
        print(object_database['M_500crit bin_indices'] == i + 1)
        bin_select = object_database['M_500crit bin_indices'] == i + 1
        plot_database = object_database[bin_select]
        max_convergence_radius = plot_database['convergence_radius'].max()

        # Plot only profiles outside the *largest* convergence radius
        radial_profiles = []
        for j in range(len(plot_database)):
            convergence_index = np.where(plot_database['bin_centre'].iloc[j] > max_convergence_radius)[0]
            radial_profiles.append(plot_database[FIELD_NAME].iloc[j][convergence_index])

        radial_profiles = np.asarray(radial_profiles)
        convergence_index = np.where(plot_database['bin_centre'].iloc[0] > max_convergence_radius)[0]
        bin_centres = plot_database['bin_centre'].iloc[0][convergence_index]
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

    # Observational data
    pratt10 = obs.Pratt10()
    bin_median, bin_perc16, bin_perc84 = pratt10.combine_entropy_profiles(
        m500_limits=(
            1e10 * unyt.Solar_Mass,
            1e17 * unyt.Solar_Mass
        ),
        k500_rescale=True
    )
    plt.fill_between(
        pratt10.radial_bins,
        bin_perc16,
        bin_perc84,
        color='aqua', alpha=0.85, linewidth=0
    )
    plt.plot(pratt10.radial_bins, bin_median, c='k')

    ax.set_xlabel(r'$R/R_{500{\rm crit}}$')
    ax.set_ylabel(plot_database.iloc[0]['ylabel'])
    ax.set_ylim([70, 2000])
    ax.set_xlim([0.02, 5])
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.legend()
    plt.title(" ".join(args.keywords))
    fig.savefig(f'{calibration_zooms.output_directory}/median_radial_profiles_{" ".join(args.keywords)}.png', dpi=300)
    if not args.quiet:
        plt.show()
    plt.close()


if __name__ == "__main__":
    results_database = utils.process_catalogue(_process_single_halo, find_keyword=args.keywords)
    results_database, bin_edges = attach_mass_bin_index(results_database)
    plot_radial_profiles_median(results_database, bin_edges)
