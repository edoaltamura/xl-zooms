"""
Test as:
    $ git pull; python3 radial_profiles_points.py -k _-8res_MinimumDistance_fixedAGNdT8.5_ -m true -r 36
"""

import os
import sys
import unyt
from matplotlib.colors import LogNorm
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
import scaling_utils as utils

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
    data.gas.subgrid_physical_densities.convert_to_physical()
    data.gas.subgrid_temperatures.convert_to_physical()

    # Select gas within sphere
    deltaX = data.gas.coordinates[:, 0] - XPotMin
    deltaY = data.gas.coordinates[:, 1] - YPotMin
    deltaZ = data.gas.coordinates[:, 2] - ZPotMin
    radial_distance = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) / R500
    index = np.where(radial_distance < 1)[0]
    del deltaX, deltaY, deltaZ

    # Calculate particle mass and rho_crit
    rho_crit = unyt.unyt_quantity(
        data.metadata.cosmology.critical_density(data.metadata.z).value,
        'g/cm**3'
    )


    number_density = (data.gas.densities / unyt.mh).to('cm**-3')
    temperature = (data.gas.temperatures).to('K')

    agn_flag = data.gas.heated_by_agnfeedback[index]
    snii_flag = data.gas.heated_by_sniifeedback[index]
    agn_flag = agn_flag > 0
    snii_flag = snii_flag > 0

    return number_density[index].value, temperature[index].value, agn_flag, snii_flag, M500, R500


@utils.set_scaling_relation_name(os.path.splitext(os.path.basename(__file__))[0])
@utils.set_output_names([
    'x',
    'y',
    'agn_flag',
    'snii_flag',
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


def plot_radial_profiles_median(object_database: pd.DataFrame) -> None:

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.loglog()

    # Bin objects by mass
    m500crit_log10 = np.array([np.log10(m.value) for m in object_database['M500'].values])
    plot_database = object_database.iloc[np.where(m500crit_log10 > 14)[0]]
    x = np.empty(0)
    y = np.empty(0)
    agn_flag = np.empty(0, dtype=np.bool)
    snii_flag = np.empty(0, dtype=np.bool)
    for j in range(len(plot_database)):
        x = np.append(x, plot_database['x'].iloc[j])
        y = np.append(y, plot_database['y'].iloc[j])
        agn_flag = np.append(agn_flag, plot_database['agn_flag'].iloc[j])
        snii_flag = np.append(snii_flag, plot_database['snii_flag'].iloc[j])

    # Set the limits of the figure.
    density_bounds = [10 ** (-7.), 1e6]  # in nh/cm^3
    temperature_bounds = [10 ** (0.3), 10 ** (9.5)]  # in K
    bins = 512

    # Make the norm object to define the image stretch
    density_bins = np.logspace(
        np.log10(density_bounds[0]), np.log10(density_bounds[1]), bins
    )
    temperature_bins = np.logspace(
        np.log10(temperature_bounds[0]), np.log10(temperature_bounds[1]), bins
    )

    H, density_edges, temperature_edges = np.histogram2d(
        x, y, bins=[density_bins, temperature_bins]
    )

    vmax = np.max(H)
    mappable = ax.pcolormesh(density_edges, temperature_edges, H.T, norm=LogNorm(vmin=1, vmax=vmax), cmap='inferno')
    fig.colorbar(mappable, ax=ax, label="Number of particles per pixel")

    H, density_edges, temperature_edges = np.histogram2d(
        x[snii_flag], y[snii_flag], bins=[density_bins, temperature_bins]
    )

    posx = np.digitize(x[snii_flag], density_edges)
    posy = np.digitize(y[snii_flag], temperature_edges)

    # select points within the histogram
    ind = (posx > 0) & (posx <= bins) & (posy > 0) & (posy <= bins)
    hhsub = H[posx[ind] - 1, posy[ind] - 1]  # values of the histogram where the points are
    x_scatter = x[snii_flag][ind][hhsub < 30]  # low density points
    y_scatter = y[snii_flag][ind][hhsub < 30]
    H[H < 30] = np.nan  # fill the areas with low density by NaNs

    ax.plot(x_scatter, y_scatter, marker=',', lw=0, linestyle="", c='lime', alpha=0.1)
    plt.contour(H.T, extent=[*density_bounds, *temperature_bounds],
                linewidths=1, color='lime', levels=[20, 200, 500])

    ax.plot(x[agn_flag], y[agn_flag], marker=',', lw=0, linestyle="", c='r', alpha=1)

    ax.set_xlabel(r"Density [$n_H$ cm$^{-3}$]")
    ax.set_ylabel(r"Temperature [K]")

    plt.legend()
    ax.set_title(
        f"$z = {calibration_zooms.redshift_from_index(args.redshift_index):.2f}$\t{''.join(args.keywords)}",
        fontsize=5
    )
    fig.savefig(
        f'{calibration_zooms.output_directory}/subgrid_density_temperature_{args.redshift_index:04d}.png',
        dpi=300
    )
    if not args.quiet:
        plt.show()
    plt.close()


if __name__ == "__main__":
    results_database = utils.process_catalogue(_process_single_halo, find_keyword=args.keywords)
    plot_radial_profiles_median(results_database)
