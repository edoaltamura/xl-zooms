"""
Test as:
    $ git pull; python3 radial_profiles_points.py -k _-8res_MinimumDistance_fixedAGNdT8.5_ -m true -r 36
"""

import os
import sys
import unyt
import argparse
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn

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

from radial_profiles import profile_3d_single_halo as profiles

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--keywords', type=str, nargs='+', required=True)
parser.add_argument('-e', '--observ-errorbars', default=False, required=False, action='store_true')
parser.add_argument('-r', '--redshift-index', type=int, default=36, required=False,
                    choices=list(range(len(calibration_zooms.get_snap_redshifts()))))
parser.add_argument('-m', '--mass-estimator', type=str.lower, default='crit', required=True,
                    choices=['crit', 'true', 'hse'])
parser.add_argument('-q', '--quiet', default=False, required=False, action='store_true')
args = parser.parse_args()

FIELD_NAME = 'mass_weighted_temps'


@utils.set_scaling_relation_name(os.path.splitext(os.path.basename(__file__))[0])
@utils.set_output_names([
    'radius',
    FIELD_NAME,
    'ylabel',
    'convergence_radius',
    'M500'
])
def _process_single_halo(zoom: Zoom):
    # Select redshift
    snapshot_file = zoom.get_redshift(args.redshift_index).snapshot_path
    catalog_file = zoom.get_redshift(args.redshift_index).catalogue_properties_path

    if args.mass_estimator == 'crit' or args.mass_estimator == 'true':

        profiles_database = profiles(snapshot_file, catalog_file, weights=FIELD_NAME)
        return profiles_database

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

        profiles_database = profiles(snapshot_file, catalog_file, weights=FIELD_NAME, hse_dataset=hse_entry)
        return profiles_database


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


def plot_radial_profiles_median(object_database: pd.DataFrame, highmass_only: bool = False) -> None:
    from matplotlib.cm import get_cmap

    name = "Set2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors = cmap.colors[3:]  # type: list

    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=colors)

    # Bin objects by mass
    m500crit_log10 = np.array([np.log10(m.value) for m in object_database['M500'].values])
    bin_log_edges = np.array([12.5, 13.4, 13.8, 14.6])
    n_bins = len(bin_log_edges) - 1
    bin_indices = np.digitize(m500crit_log10, bin_log_edges)

    # Display zoom data
    for i in range(n_bins if highmass_only else 1, n_bins + 1):

        plot_database = object_database.iloc[np.where(bin_indices == i)[0]]
        max_convergence_radius = plot_database['convergence_radius'].max()

        # Plot only profiles outside the *largest* convergence radius
        radius = np.empty(0)
        field = np.empty(0)
        for j in range(len(plot_database)):
            # convergence_index = np.where(plot_database['radius'].iloc[j] > max_convergence_radius)[0]
            # radius = np.append(radius, plot_database['radius'].iloc[j][convergence_index])
            # field = np.append(field, plot_database[FIELD_NAME].iloc[j][convergence_index])

            radius = np.append(radius, plot_database['radius'].iloc[j])
            field = np.append(field, plot_database[FIELD_NAME].iloc[j])

        ax.plot(radius[::2], field[::2], marker=',', lw=0, linestyle="", c=colors[i - 1], alpha=0.9)


    # Display observational data
    observations_color = (0.65, 0.65, 0.65)

    # Observational data
    # pratt10 = obs.Pratt10()
    # bin_median, bin_perc16, bin_perc84 = pratt10.combine_entropy_profiles(
    #     m500_limits=(
    #         10 ** bin_log_edges[-2] * unyt.Solar_Mass,
    #         10 ** bin_log_edges[-1] * unyt.Solar_Mass,
    #     ),
    #     k500_rescale=True
    # )
    # plt.fill_between(
    #     pratt10.radial_bins,
    #     bin_perc16,
    #     bin_perc84,
    #     color=observations_color,
    #     alpha=0.8,
    #     linewidth=0
    # )
    # plt.plot(
    #     pratt10.radial_bins, bin_median, c='k',
    #     label=(
    #         f"{pratt10.citation:s} ($10^{{{bin_log_edges[i - 1]:.1f}}}"
    #         f"<M_{{500}}/M_{{\odot}}<10^{{{bin_log_edges[i]:.1f}}}$)"
    #     )
    # )
    # del pratt10

    ax.set_xlabel(f'$R/R_{{500,{args.mass_estimator}}}$')
    ax.set_ylabel(plot_database.iloc[0]['ylabel'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([0.01, 2.5])
    ax.set_ylim([1e5, 1e10])
    plt.legend()
    ax.set_title(
        f"$z = {calibration_zooms.redshift_from_index(args.redshift_index):.2f}$\t{''.join(args.keywords)}",
        fontsize=5
    )
    fig.savefig(
        f'{calibration_zooms.output_directory}/nobins_radial_profiles_{args.redshift_index:04d}.png',
        dpi=300
    )
    if not args.quiet:
        plt.show()
    plt.close()


if __name__ == "__main__":
    results_database = utils.process_catalogue(_process_single_halo, find_keyword=args.keywords)
    plot_radial_profiles_median(results_database, highmass_only=True)
