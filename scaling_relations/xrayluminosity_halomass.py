"""
Plot scaling relations for EAGLE-XL tests

Run using:
    git pull; python3 xrayluminosity_halomass.py -r 36 -m true -k dT8_ dT8.5_
"""
import sys
import os
import unyt
import argparse
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import OrderedDict
import swiftsimio as sw
import velociraptor as vr

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

# Make the register backend visible to the script
sys.path.append("../zooms")
sys.path.append("../observational_data")
sys.path.append("../hydrostatic_estimates")
sys.path.append("../xray")

# Import backend utilities
from register import zooms_register, Tcut_halogas, calibration_zooms, Zoom
import observational_data as obs
import scaling_utils as utils
import scaling_style as style

# Import modules for calculating additional quantities
from relaxation import process_single_halo as relaxation_index
from cloudy_softband import interpolate_X_Ray, logsumexp

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--keywords', type=str, nargs='+', required=True)
parser.add_argument('-e', '--observ-errorbars', default=True, required=False, action='store_true')
parser.add_argument('-r', '--redshift-index', type=int, default=36, required=False,
                    choices=list(range(len(calibration_zooms.get_snap_redshifts()))))
parser.add_argument('-m', '--mass-estimator', type=str.lower, default='crit', required=True,
                    choices=['crit', 'true', 'hse'])
parser.add_argument('-q', '--quiet', default=False, required=False, action='store_true')
args = parser.parse_args()

core_excised: bool = False


def process_single_halo(
        path_to_snap: str,
        path_to_catalogue: str,
        hse_dataset: pd.Series = None,
) -> Tuple[unyt.unyt_quantity]:
    _, kinetic_energy, thermal_energy = relaxation_index(
        path_to_snap,
        path_to_catalogue
    )
    relaxed = kinetic_energy / thermal_energy

    # Read in halo properties
    vr_catalogue_handle = vr.load(path_to_catalogue)
    a = vr_catalogue_handle.a

    if vr_catalogue_handle.z > 1e-5:
        raise ValueError(
            f"The current CLOUDY tables only support redshift 0. "
            f"Detected z = {vr_catalogue_handle.z}"
        )

    M500 = vr_catalogue_handle.spherical_overdensities.mass_500_rhocrit[0].to('Msun')
    R500 = vr_catalogue_handle.spherical_overdensities.r_500_rhocrit[0].to('Mpc')
    XPotMin = vr_catalogue_handle.positions.xcminpot[0].to('Mpc')
    YPotMin = vr_catalogue_handle.positions.ycminpot[0].to('Mpc')
    ZPotMin = vr_catalogue_handle.positions.zcminpot[0].to('Mpc')

    # If no custom aperture, select R500c as default
    if hse_dataset is not None:
        assert R500.units == hse_dataset["R500hse"].units
        assert M500.units == hse_dataset["M500hse"].units
        R500 = hse_dataset["R500hse"]
        M500 = hse_dataset["M500hse"]

    # Apply spatial mask to particles. SWIFTsimIO needs comoving coordinates
    # to filter particle coordinates, while VR outputs are in physical units.
    # Convert the region bounds to comoving, but keep the CoP and Rcrit in
    # physical units for later use.
    mask = sw.mask(path_to_snap, spatial_only=True)
    region = [
        [(XPotMin - 0.5 * R500) * a, (XPotMin + 0.5 * R500) * a],
        [(YPotMin - 0.5 * R500) * a, (YPotMin + 0.5 * R500) * a],
        [(ZPotMin - 0.5 * R500) * a, (ZPotMin + 0.5 * R500) * a]
    ]
    mask.constrain_spatial(region)
    data = sw.load(path_to_snap, mask=mask)

    # Convert datasets to physical quantities
    # R500c is already in physical units
    data.gas.coordinates.convert_to_physical()
    data.gas.masses.convert_to_physical()
    data.gas.temperatures.convert_to_physical()
    data.gas.densities.convert_to_physical()

    # Select hot gas within sphere and without core
    tempGas = data.gas.temperatures
    deltaX = data.gas.coordinates[:, 0] - XPotMin
    deltaY = data.gas.coordinates[:, 1] - YPotMin
    deltaZ = data.gas.coordinates[:, 2] - ZPotMin
    deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) / R500

    # Keep only particles inside R500crit
    if core_excised:
        index = np.where((deltaR > 0.15) & (deltaR < 1) & (tempGas > 1e5))[0]
    else:
        index = np.where((deltaR < 1) & (tempGas > 1e5))[0]

    del tempGas, deltaX, deltaY, deltaZ, deltaR, a

    # Compute hydrogen number density and the log10
    # of the temperature to provide to the xray interpolator.
    data_nH = np.log10(data.gas.element_mass_fractions.hydrogen * data.gas.densities.to('g*cm**-3') / unyt.mp)
    data_T = np.log10(data.gas.temperatures.value)

    # Interpolate the Cloudy table to get emissivities
    emissivities = interpolate_X_Ray(
        data_nH,
        data_T,
        data.gas.element_mass_fractions
    )
    # emissivities = unyt.unyt_array((10 ** emissivities.astype(np.float32)).astype(np.float64), 'erg/s/cm**3')
    # emissivities = emissivities.to('erg/s/Mpc**3')

    # Compute X-ray luminosities
    # LX = emissivity * gas_mass / gas_density
    # xray_luminosities = emissivities[index] * (data.gas.masses[index] / data.gas.densities[index]).to('Mpc**3')
    # xray_luminosities[~np.isfinite(xray_luminosities)] = 0

    log10_Mpc3_to_cm3 = np.log10(unyt.Mpc.get_conversion_factor(unyt.cm)[0] ** 3)

    LX = unyt.unyt_quantity(
        10 ** (
                logsumexp(
                    emissivities[index],
                    b=(data.gas.masses[index] / data.gas.densities[index]).value,
                    base=10.
                ) + log10_Mpc3_to_cm3
        ), 'erg/s'
    )

    assert LX.units == unyt.erg / unyt.s

    return M500, LX, relaxed


@utils.set_scaling_relation_name(os.path.splitext(os.path.basename(__file__))[0])
@utils.set_output_names([
    'M500',
    'LX',
    'Ekin/Eth'
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


def mass_xray_luminosity(results: pd.DataFrame):
    fig, ax = plt.subplots()
    legend_handles = []
    for i in range(len(results)):

        run_style = style.get_style_for_object(results.loc[i, "Run name"])
        if run_style['Legend handle'] not in legend_handles:
            legend_handles.append(run_style['Legend handle'])
        legend_handles = list(OrderedDict.fromkeys(legend_handles))

        ax.scatter(
            results.loc[i, "M500"],
            results.loc[i, "LX"],
            marker=run_style['Marker style'],
            s=run_style['Marker size'],
            edgecolors=run_style['Color'] if results.loc[i, "Ekin/Eth"].value > 0.1 else 'none',
            facecolors='w' if results.loc[i, "Ekin/Eth"].value > 0.1 else run_style['Color'],
            linewidth=0.4 if results.loc[i, "Ekin/Eth"].value > 0.1 else 0,
        )

    # Build legends
    legend_sims = plt.legend(handles=legend_handles, loc=2, frameon=True, facecolor='w', edgecolor='none')
    ax.add_artist(legend_sims)

    # Display observational data
    observations_color = (0.65, 0.65, 0.65)
    handles = []

    Pratt10 = obs.Pratt10()
    ax.scatter(Pratt10.M500, Pratt10.LX_R500,
               marker='D', s=5, alpha=1, color=observations_color, edgecolors='none', zorder=0)
    if args.observ_errorbars:
        ax.errorbar(Pratt10.M500, Pratt10.LX_R500,
                    yerr=(Pratt10.Delta_lo_LX_R500, Pratt10.Delta_hi_LX_R500),
                    xerr=(Pratt10.Delta_lo_M500, Pratt10.Delta_hi_M500),
                    ls='none', elinewidth=0.5, color=observations_color, zorder=0)
    handles.append(
        Line2D([], [], color=observations_color, marker='D', markeredgecolor='none', linestyle='None', markersize=4,
               label=Pratt10.citation)
    )
    del Pratt10

    Barnes17 = obs.Barnes17().hdf5.z000p101.SPEC
    relaxed = Barnes17.Ekin_500 / Barnes17.Ethm_500
    ax.scatter(Barnes17.M500[relaxed < 0.1], Barnes17.LXbol_500[relaxed < 0.1],
               marker='s', s=6, alpha=1, color='k', edgecolors='none', zorder=0)
    ax.scatter(Barnes17.M500[relaxed > 0.1], Barnes17.LXbol_500[relaxed > 0.1],
               marker='s', s=5, alpha=1, facecolors='w', edgecolors='k', linewidth=0.4, zorder=0)
    handles.append(
        Line2D([], [], color='k', marker='s', markeredgecolor='none', linestyle='None', markersize=4,
               label=obs.Barnes17().citation + ' $z=0.1$')
    )
    del Barnes17

    ax.axhspan(0.407e44, 20e44, facecolor='lime', alpha=0.2)

    ax.set_xlabel(f'$M_{{500,{{\\rm {args.mass_estimator}}}}}\\ [{{\\rm M}}_{{\\odot}}]$')
    ax.set_ylabel(f'$M_{{500,{{\\rm gas, {args.mass_estimator}}}}}\\ [{{\\rm M}}_{{\\odot}}]$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f"$z = {calibration_zooms.redshift_from_index(args.redshift_index):.2f}$")
    fig.savefig(f'{calibration_zooms.output_directory}/m500{args.mass_estimator}_hotgas_{args.redshift_index:d}.png',
                dpi=300)
    if not args.quiet:
        plt.show()
    plt.close()


if __name__ == "__main__":
    results = utils.process_catalogue(_process_single_halo, find_keyword=args.keywords)
    mass_xray_luminosity(results)
