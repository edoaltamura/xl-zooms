"""
Test as:
    $ git pull; python3 radial_profiles_points.py -k _-8res_MinimumDistance_fixedAGNdT8.5_ -m true -r 36
"""

import os
import sys
import unyt
import h5py as h5
import numpy as np
import pandas as pd
import swiftsimio as sw
import velociraptor as vr
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
from auto_parser import args
import observational_data as obs
import scaling_utils as utils
from convergence_radius import convergence_radius

# Constants
mean_molecular_weight = 0.59
mean_atomic_weight_per_free_electron = 1.14


def profile_3d_single_halo(
        path_to_snap: str,
        path_to_catalogue: str,
        hse_dataset: pd.Series = None,
) -> tuple:

    # Read in halo properties
    vr_catalogue_handle = vr.load(path_to_catalogue)
    a = vr_catalogue_handle.a
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
        [(XPotMin - R500) * a, (XPotMin + R500) * a],
        [(YPotMin - R500) * a, (YPotMin + R500) * a],
        [(ZPotMin - R500) * a, (ZPotMin + R500) * a]
    ]
    mask.constrain_spatial(region)
    data = sw.load(path_to_snap, mask=mask)

    # Convert datasets to physical quantities
    # R500c is already in physical units
    data.gas.coordinates.convert_to_physical()
    data.gas.masses.convert_to_physical()
    data.gas.temperatures.convert_to_physical()
    data.gas.densities.convert_to_physical()

    # Select hot gas within sphere
    tempGas = data.gas.temperatures
    deltaX = data.gas.coordinates[:, 0] - XPotMin
    deltaY = data.gas.coordinates[:, 1] - YPotMin
    deltaZ = data.gas.coordinates[:, 2] - ZPotMin
    radial_distance = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) / R500
    index = np.where((radial_distance < 2) & (tempGas > 1e5))[0]
    del tempGas, deltaX, deltaY, deltaZ

    # Calculate particle mass and rho_crit
    rho_crit = unyt.unyt_quantity(
        data.metadata.cosmology.critical_density(data.metadata.z).value,
        'g/cm**3'
    )

    radial_distance = radial_distance[index]
    data.gas.densities = data.gas.densities[index]
    data.gas.masses = data.gas.masses[index]
    data.gas.temperatures = data.gas.temperatures[index]

    data.gas.mass_weighted_temperatures = data.gas.masses * data.gas.temperatures * unyt.boltzmann_constant
    data.gas.number_densities = (data.gas.densities.to('g/cm**3') / (unyt.mp * mean_molecular_weight)).to('cm**-3')
    field_value = data.gas.mass_weighted_temperatures / data.gas.number_densities ** (2 / 3)
    field_value = field_value.to('keV*cm**2')
    field_label = r'$K$ [keV cm$^2$]'

    return radial_distance, field_value, field_label, M500, R500


@utils.set_scaling_relation_name(os.path.splitext(os.path.basename(__file__))[0])
@utils.set_output_names([
    'radial_distance',
    'field_value',
    'field_label',
    'M500',
    'R500'
])
def _process_single_halo(zoom: Zoom):
    # Select redshift
    snapshot_file = zoom.get_redshift(args.redshift_index).snapshot_path
    catalog_file = zoom.get_redshift(args.redshift_index).catalogue_properties_path

    if args.mass_estimator == 'crit' or args.mass_estimator == 'true':

        profiles_database = profile_3d_single_halo(snapshot_file, catalog_file)
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

        profiles_database = profile_3d_single_halo(snapshot_file, catalog_file, hse_dataset=hse_entry)
        return profiles_database


def plot_radial_profiles_median(object_database: pd.DataFrame, highmass_only: bool = False) -> None:
    # kBT200 = (unyt.G * mean_molecular_weight * M200c * unyt.mass_proton / 2 / R200c).to('keV')
    # K200 = (kBT200 / (3 * M200c * obs.cosmic_fbary / (4 * np.pi * R200c ** 3 * unyt.mass_proton)) ** (2 / 3)).to(
    #     'keV*cm**2')
    # print('Virial temperature = ', kBT200)
    #
    # number_density_gas = data.gas.densities / (mean_molecular_weight * unyt.mass_proton)
    # number_density_gas = number_density_gas.to('1/cm**3')
    #

    fig, ax = plt.subplots()

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
            radius = np.append(radius, plot_database['radial_distance'].iloc[j])
            field = np.append(field, plot_database['field_value'].iloc[j])

        ax.plot(radius[::2], field[::2], marker=',', lw=0, linestyle="", c='orange', alpha=0.9)

    # Display observational data
    observations_color = (0.65, 0.65, 0.65)

    # Observational data
    pratt10 = obs.Pratt10()
    bin_median, bin_perc16, bin_perc84 = pratt10.combine_entropy_profiles(
        m500_limits=(
            10 ** bin_log_edges[-2] * unyt.Solar_Mass,
            10 ** bin_log_edges[-1] * unyt.Solar_Mass,
        ),
        k500_rescale=True
    )
    plt.fill_between(
        pratt10.radial_bins,
        bin_perc16,
        bin_perc84,
        color=observations_color,
        alpha=0.8,
        linewidth=0
    )
    plt.plot(
        pratt10.radial_bins, bin_median, c='k',
        label=(
            f"{pratt10.citation:s} ($10^{{{bin_log_edges[i - 1]:.1f}}}"
            f"<M_{{500}}/M_{{\odot}}<10^{{{bin_log_edges[i]:.1f}}}$)"
        )
    )
    del pratt10

    ax.set_xlabel(f'$R/R_{{500,{args.mass_estimator}}}$')
    ax.set_ylabel(plot_database.iloc[0]['field_label'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_xlim([0.01, 2.5])
    # ax.set_ylim([1e5, 1e10])
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
