# Plot scaling relations for EAGLE-XL tests
import sys
import os
import unyt
import argparse
import h5py as h5
import numpy as np
import pandas as pd
import swiftsimio as sw
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

# Make the register backend visible to the script
sys.path.append("../../zooms")
sys.path.append("../observational_data")

from register import zooms_register, Zoom, Tcut_halogas, calibration_zooms
import observational_data as obs
import scaling_utils as utils
import scaling_style as style


parser = argparse.ArgumentParser()
parser.add_argument('-k', '--keywords', type=str, nargs='+', required=True)
parser.add_argument('-e', '--observ-errorbars', default=False, required=False, action='store_true')
parser.add_argument('-r', '--redshift-index', type=int, default=37, required=False,
                    choices=list(range(len(calibration_zooms.get_snap_redshifts()))))
parser.add_argument('-m', '--mass-estimator', type=str.lower, default='crit', required=True,
                    choices=['crit', 'true', 'hse'])
parser.add_argument('-q', '--quiet', default=False, required=False, action='store_true')
args = parser.parse_args()


def process_single_halo(
        path_to_snap: str,
        path_to_catalogue: str,
        hse_dataset: pd.Series = None,
) -> Tuple[unyt.unyt_quantity]:
    # Read in halo properties
    with h5.File(path_to_catalogue, 'r') as h5file:
        scale_factor = float(h5file['/SimulationInfo'].attrs['ScaleFactor'])
        M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
        R500c = unyt.unyt_quantity(h5file['/SO_R_500_rhocrit'][0], unyt.Mpc) / scale_factor
        XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc) / scale_factor
        YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc) / scale_factor
        ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc) / scale_factor

        # If no custom aperture, select R500c as default
        if hse_dataset is not None:
            assert R500c.units == hse_dataset["R500hse"].units
            assert M500c.units == hse_dataset["M500hse"].units
            R500c = hse_dataset["R500hse"]
            M500c = hse_dataset["M500hse"]

    # Read in gas particles to compute the core-excised temperature
    mask = sw.mask(path_to_snap, spatial_only=False)
    region = [[XPotMin - 1.1 * R500c, XPotMin + 1.1 * R500c],
              [YPotMin - 1.1 * R500c, YPotMin + 1.1 * R500c],
              [ZPotMin - 1.1 * R500c, ZPotMin + 1.1 * R500c]]
    mask.constrain_spatial(region)
    mask.constrain_mask(
        "gas", "temperatures",
        Tcut_halogas * mask.units.temperature,
        1.e12 * mask.units.temperature
    )
    data = sw.load(path_to_snap, mask=mask)
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

    # Select redshift
    snapshot_file = zoom.get_redshift(args.redshift_index).snapshot_path
    catalog_file = zoom.get_redshift(args.redshift_index).catalogue_properties_path

    if args.mass_estimator == 'crit' or args.mass_estimator == 'true':

        return process_single_halo(snapshot_file, catalog_file)

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

        return process_single_halo(snapshot_file, catalog_file, hse_dataset=hse_entry)


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

    if args.mass_estimator == 'crit' or args.mass_estimator == 'true':
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
    ax.axhline(y=0.1, linestyle='--', linewidth=1, color='k')
    ax.set_xlabel(f'$M_{{500,{{\\rm {args.mass_estimator}}}}}\\ [{{\\rm M}}_{{\\odot}}]$')
    ax.set_ylabel(r'$E_{\rm kin}\ /E_{\rm therm}\ (<R_{500{\rm crit}})$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([0.02, 1.5])
    ax.set_title(f"$z = {calibration_zooms.redshift_from_index(args.redshift_index):.2f}$")
    fig.savefig(f'{calibration_zooms.output_directory}/relaxation_{args.redshift_index:d}.png', dpi=300)
    if not args.quiet:
        plt.show()
    plt.close()


if __name__ == "__main__":
    results = utils.process_catalogue(_process_single_halo, find_keyword=args.keywords)
    relaxation(results)
