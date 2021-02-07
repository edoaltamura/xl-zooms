# Plot scaling relations for EAGLE-XL tests
import sys
import os
import unyt
import numpy as np
from typing import Tuple
import h5py as h5
import swiftsimio as sw
import pandas as pd
import matplotlib.pyplot as plt
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
        XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc)
        YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc)
        ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc)

    # Read in gas particles to compute the core-excised temperature
    mask = sw.mask(f'{path_to_snap}', spatial_only=False)
    region = [[XPotMin - 1.1 * R500c, XPotMin + 1.1 * R500c],
              [YPotMin - 1.1 * R500c, YPotMin + 1.1 * R500c],
              [ZPotMin - 1.1 * R500c, ZPotMin + 1.1 * R500c]]
    mask.constrain_spatial(region)
    mask.constrain_mask(
        "gas", "temperatures",
        Tcut_halogas * mask.units.temperature,
        1.e12 * mask.units.temperature
    )
    data = sw.load(f'{path_to_snap}', mask=mask)
    posGas = data.gas.coordinates
    massGas = data.gas.masses
    velGas = data.gas.velocities
    mass_weighted_temperatures = data.gas.temperatures * data.gas.masses

    # Select hot gas within sphere and without core
    deltaX = posGas[:, 0] - XPotMin
    deltaY = posGas[:, 1] - YPotMin
    deltaZ = posGas[:, 2] - ZPotMin
    deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2)

    # Count only particles inside R500crit
    index = np.where(deltaR < R500c)[0]

    # Compute kinetic energy in the halo's rest frame
    peculiar_velocity = np.sum(velGas[index] * massGas[index, None], axis=0) / np.sum(massGas[index])
    velGas[:, 0] -= peculiar_velocity[0]
    velGas[:, 1] -= peculiar_velocity[1]
    velGas[:, 2] -= peculiar_velocity[2]

    Ekin = np.sum(
        0.5 * massGas[index] * (velGas[index, 0] ** 2 + velGas[index, 1] ** 2 + velGas[index, 2] ** 2)
    ).to("1.e10*Mpc**2*Msun/Gyr**2")
    Etherm = np.sum(
        1.5 * unyt.boltzmann_constant * mass_weighted_temperatures[index] / (unyt.hydrogen_mass / 1.16)
    ).to("1.e10*Mpc**2*Msun/Gyr**2")

    return M500c, Ekin, Etherm


@utils.set_scaling_relation_name(os.path.splitext(os.path.basename(__file__))[0])
@utils.set_output_names([
    'M_500crit',
    'Ekin_500crit',
    'Etherm_500crit'
])
def _process_single_halo(zoom: Zoom):
    return process_single_halo(zoom.snapshot_file, zoom.catalog_file)


def relaxation(results: pd.DataFrame):
    fig, ax = plt.subplots()
    legend_handles = []
    for i in range(len(results)):

        run_style = style.get_style_for_object(results.loc[i, "Run name"])
        if run_style['Legend handle'] not in legend_handles:
            legend_handles.append(run_style['Legend handle'])

        energy_ratio = results.loc[i, "Ekin_500crit"] / results.loc[i, "Etherm_500crit"]

        ax.scatter(
            results.loc[i, "M_500crit"],
            energy_ratio,
            marker=run_style['Marker style'],
            s=run_style['Marker size'],
            alpha=1,
            edgecolors=run_style['Color'] if energy_ratio.value > 0.1 else 'none',
            facecolors='w'if energy_ratio.value > 0.1 else run_style['Color'],
            linewidth=0.4 if energy_ratio.value > 0.1 else 0,
            zorder=5
        )

    # Build legends
    legend_sims = plt.legend(handles=legend_handles, loc=2)
    ax.add_artist(legend_sims)

    # Display observational data
    handles = []

    Barnes17 = obs.Barnes17().hdf5.z000p000.true
    relaxed = Barnes17.Ekin_500 / Barnes17.Ethm_500
    ax.scatter(Barnes17.M500[relaxed < 0.1], relaxed[relaxed < 0.1],
               marker='s', s=6, alpha=1, color='k', edgecolors='none', zorder=0)
    ax.scatter(Barnes17.M500[relaxed > 0.1], relaxed[relaxed > 0.1],
               marker='s', s=5, alpha=1, facecolors='w', edgecolors='k', linewidth=0.4, zorder=0)
    handles.append(
        Line2D([], [], color='k', marker='s', markeredgecolor='none', linestyle='None', markersize=4,
               label=obs.Barnes17().citation + ' $z=0$')
    )
    del Barnes17

    legend_obs = plt.legend(handles=handles, loc=4, frameon=True, facecolor='w', edgecolor='none')
    ax.add_artist(legend_obs)
    ax.axhline(y=0.1, linestyle='--')
    ax.set_xlabel(r'$M_{500{\rm crit}}\ [{\rm M}_{\odot}]$')
    ax.set_ylabel(r'$E_{\rm kin}\ /E_{\rm therm}\ (R_{500{\rm crit}})$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{zooms_register[0].output_directory}/relaxation.png', dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    import sys

    if sys.argv[1]:
        keyword = sys.argv[1]
    else:
        keyword = 'Ref'

    results = utils.process_catalogue(_process_single_halo, find_keyword=keyword)
    relaxation(results)
