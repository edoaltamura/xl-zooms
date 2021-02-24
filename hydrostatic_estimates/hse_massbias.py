# Plot scaling relations for EAGLE-XL tests
import sys
import os
import unyt
import numpy as np
from typing import Tuple
import h5py as h5
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Make the register backend visible to the script
sys.path.append("../zooms")
sys.path.append("../observational_data")
sys.path.append("../scaling_relations")

from register import zooms_register, Zoom, Tcut_halogas, name_list
from vikhlinin_hse import HydrostaticEstimator
import observational_data as obs
import scaling_utils as utils
import scaling_style as style

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

cosmology = obs.Observations().cosmo_model
fbary = cosmology.Ob0 / cosmology.Om0  # Cosmic baryon fraction

plot_observation_errorbars = False


def process_single_halo(
        path_to_snap: str,
        path_to_catalogue: str
) -> Tuple[unyt.unyt_quantity]:
    true_hse = HydrostaticEstimator.from_data_paths(
        catalog_file=path_to_catalogue,
        snapshot_file=path_to_snap,
        profile_type='true',
        diagnostics_on=True
    )

    true_hse.plot_diagnostics()

    output = (
        true_hse.R500c,
        true_hse.R200hse,
        true_hse.R500hse,
        true_hse.R2500hse,
        true_hse.M500c,
        true_hse.M200hse,
        true_hse.M500hse,
        true_hse.M2500hse,
        true_hse.ne200hse,
        true_hse.ne500hse,
        true_hse.ne2500hse,
        true_hse.kBT200hse,
        true_hse.kBT500hse,
        true_hse.kBT2500hse,
        true_hse.P200hse,
        true_hse.P500hse,
        true_hse.P2500hse,
        true_hse.K200hse,
        true_hse.K500hse,
        true_hse.K2500hse,
        true_hse.b200hse,
        true_hse.b500hse,
        true_hse.b2500hse
    )

    return output


@utils.set_scaling_relation_name(os.path.splitext(os.path.basename(__file__))[0])
@utils.set_output_names([
    'R500c',
    'R200hse',
    'R500hse',
    'R2500hse',
    'M500c',
    'M200hse',
    'M500hse',
    'M2500hse',
    'ne200hse',
    'ne500hse',
    'ne2500hse',
    'kBT200hse',
    'kBT500hse',
    'kBT2500hse',
    'P200hse',
    'P500hse',
    'P2500hse',
    'K200hse',
    'K500hse',
    'K2500hse',
    'b200hse',
    'b500hse',
    'b2500hse'
])
def _process_single_halo(zoom: Zoom):
    return process_single_halo(zoom.snapshot_file, zoom.catalog_file)


def true_mass_bias(results: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.axhline(y=1, linestyle='--', linewidth=1, color='k')
    legend_handles = []
    for i in range(len(results)):

        run_style = style.get_style_for_object(results.loc[i, "Run name"])
        if run_style['Legend handle'] not in legend_handles:
            legend_handles.append(run_style['Legend handle'])

        ax.scatter(
            results.loc[i, "M500c"],
            results.loc[i, "M500hse"] / results.loc[i, "M500c"],
            marker=run_style['Marker style'],
            c=run_style['Color'],
            s=run_style['Marker size'],
            alpha=1,
            edgecolors='none',
            zorder=5
        )

    # Build legends
    legend_sims = plt.legend(handles=legend_handles, loc=2, frameon=True, facecolor='w', edgecolor='none')
    ax.add_artist(legend_sims)

    # Display observational data
    observations_color = (0.65, 0.65, 0.65)
    handles = []

    Barnes17 = obs.Barnes17()
    relaxed = Barnes17.hdf5.z000p101.true.Ekin_500 / Barnes17.hdf5.z000p101.true.Ethm_500
    ax.scatter(Barnes17.m_500true[relaxed < 0.1], Barnes17.m_500hse[relaxed < 0.1] / Barnes17.m_500true[relaxed < 0.1],
               marker='s', s=6, alpha=1, color=observations_color, edgecolors='none', zorder=0)
    ax.scatter(Barnes17.m_500true[relaxed > 0.1], Barnes17.m_500hse[relaxed > 0.1] / Barnes17.m_500true[relaxed > 0.1],
               marker='s', s=5, alpha=1, facecolors='w', edgecolors=observations_color, linewidth=0.4, zorder=0)
    handles.append(
        Line2D([], [], color=observations_color, marker='s', markeredgecolor='none', linestyle='None', markersize=4,
               label=obs.Barnes17().citation + ' $z=0.1$')
    )
    del Barnes17

    legend_obs = plt.legend(handles=handles, loc=4, frameon=True, facecolor='w', edgecolor='none')
    ax.add_artist(legend_obs)
    ax.set_xlabel(r'$M_{500{\rm true}}\ [{\rm M}_{\odot}]$')
    ax.set_ylabel(r'$M_{500{\rm hse}} / M_{500{\rm true}}$')
    ax.set_xscale('log')
    # ax.set_ylim([-0.07, 0.27])
    ax.set_xlim([4e12, 6e15])

    fig.savefig(f'{zooms_register[0].output_directory}/mass_bias_true.png', dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    results = utils.process_catalogue(_process_single_halo,
                                      find_keyword='dT8',
                                      save_dataframe=True,
                                      asynchronous_threading=True)
    true_mass_bias(results)
