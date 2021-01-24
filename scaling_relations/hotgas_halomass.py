# Plot scaling relations for EAGLE-XL tests
import sys
import os
import unyt
import numpy as np
from typing import Tuple
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
import scaling_utils as utils
import scaling_style as style

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
        M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
        R500c = unyt.unyt_quantity(h5file['/SO_R_500_rhocrit'][0], unyt.Mpc)

        # If the hot gas mass already computed by VR, use that instead of calculating it
        Mhot500c_key = "SO_Mass_gas_highT_1.000000_times_500.000000_rhocrit"

        if Mhot500c_key in h5file.keys():
            Mhot500c = unyt.unyt_quantity(h5file[f'/{Mhot500c_key}'][0] * 1.e10, unyt.Solar_Mass)

        else:
            XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc)
            YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc)
            ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc)

            import swiftsimio as sw

            # Read in gas particles
            mask = sw.mask(f'{path_to_snap}', spatial_only=False)
            region = [[XPotMin - R500c, XPotMin + R500c],
                      [YPotMin - R500c, YPotMin + R500c],
                      [ZPotMin - R500c, ZPotMin + R500c]]
            mask.constrain_spatial(region)
            mask.constrain_mask(
                "gas", "temperatures",
                Tcut_halogas * mask.units.temperature,
                1.e12 * mask.units.temperature
            )
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
            Mhot500c = Mhot500c.to(unyt.Solar_Mass)

    return M500c, Mhot500c, Mhot500c / M500c


@utils.set_scaling_relation_name(os.path.splitext(os.path.basename(__file__))[0])
@utils.set_output_names([
    'M_500crit',
    'Mhot500c',
    'fhot500c'
])
def _process_single_halo(zoom: Zoom):
    return process_single_halo(zoom.snapshot_file, zoom.catalog_file)


def m_500_hotgas(results: pd.DataFrame):
    fig, ax = plt.subplots()
    legend_handles = []
    for i in range(len(results)):

        run_style = style.get_style_for_object(results.loc[i, "Run name"])
        if run_style['Legend handle'] not in legend_handles:
            legend_handles.append(run_style['Legend handle'])

        ax.scatter(
            results.loc[i, "M_500crit"],
            results.loc[i, "Mhot500c"],
            marker=run_style['Marker style'],
            c=run_style['Color'],
            alpha=0.7,
            s=run_style['Marker size'],
            edgecolors='none', zorder=5
        )

    # Build legends
    legend_sims = plt.legend(handles=legend_handles, loc=2)

    # Display observational data
    Sun09 = obs.Sun09()
    Lovisari15 = obs.Lovisari15()
    ax.scatter(Sun09.M500, Sun09.Mgas500, marker='d', s=8, alpha=1,
               color=(0.65, 0.65, 0.65), edgecolors='none', zorder=0)
    ax.scatter(Lovisari15.M500, Lovisari15.Mgas500, marker='s', s=8, alpha=1,
               color=(0.65, 0.65, 0.65), edgecolors='none', zorder=0)

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


if __name__ == "__main__":
    results = utils.process_catalogue(_process_single_halo, find_keyword='Ref')
    m_500_hotgas(results)