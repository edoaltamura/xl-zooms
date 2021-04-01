"""
Test as:
    $ git pull; python3 temperature_density.py -k VR18_-8res_MinimumDistance_fixedAGNdT8.5_ -m true -r 36
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
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

# Make the register backend visible to the script
sys.path.append("../observational_data")
sys.path.append("../scaling_relations")
sys.path.append("../zooms")

from register import zooms_register, Zoom, Tcut_halogas, calibration_zooms
from auto_parser import args, parser
import scaling_utils as utils

parser.add_argument(
    '-a',
    '--aperture-percent',
    type=int,
    default=100,
    required=False,
    choices=list(range(5, 300))
)
args = parser.parse_args()

# Constants
aperture_fraction = args.aperture_percent / 100
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
        [(XPotMin - aperture_fraction * R500) / a, (XPotMin + aperture_fraction * R500) / a],
        [(YPotMin - aperture_fraction * R500) / a, (YPotMin + aperture_fraction * R500) / a],
        [(ZPotMin - aperture_fraction * R500) / a, (ZPotMin + aperture_fraction * R500) / a]
    ]
    mask.constrain_spatial(region)
    data = sw.load(path_to_snap, mask=mask)

    # Convert datasets to physical quantities
    # R500c is already in physical units
    data.gas.coordinates.convert_to_physical()
    data.gas.masses.convert_to_physical()
    data.gas.temperatures.convert_to_physical()
    data.gas.densities.convert_to_physical()

    # Select gas within sphere and main FOF halo
    fof_id = data.gas.fofgroup_ids
    deltaX = data.gas.coordinates[:, 0] - XPotMin
    deltaY = data.gas.coordinates[:, 1] - YPotMin
    deltaZ = data.gas.coordinates[:, 2] - ZPotMin
    radial_distance = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) / R500
    index = np.where((radial_distance < aperture_fraction) & (fof_id == 1))[0]
    del deltaX, deltaY, deltaZ

    number_density = (data.gas.densities / unyt.mh).to('cm**-3').value[index]
    temperature = (data.gas.temperatures).to('K').value[index]

    agn_flag = data.gas.heated_by_agnfeedback[index]
    snii_flag = data.gas.heated_by_sniifeedback[index]
    agn_flag = agn_flag > 0
    snii_flag = snii_flag > 0

    # Calculate the critical density for the cross-hair marker
    rho_crit = unyt.unyt_quantity(
        data.metadata.cosmology.critical_density(data.metadata.z).value, 'g/cm**3'
    ).to('Msun/Mpc**3')
    nH_500 = (rho_crit * 500 / unyt.mh).to('cm**-3')

    return number_density, temperature, agn_flag, snii_flag, M500, R500, nH_500


@utils.set_scaling_relation_name(os.path.splitext(os.path.basename(__file__))[0])
@utils.set_output_names([
    'x',
    'y',
    'agn_flag',
    'snii_flag',
    'M500',
    'R500',
    'nH_500'
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


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if f < 1e-2:
        float_str = "{0:.0e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def draw_adiabats(axes, density_bins, temperature_bins):
    density_interps, temperature_interps = np.meshgrid(density_bins, temperature_bins)
    temperature_interps *= unyt.K * unyt.boltzmann_constant
    entropy_interps = temperature_interps / (density_interps / unyt.cm ** 3) ** (2 / 3)
    entropy_interps = entropy_interps.to('keV*cm**2').value

    # Define entropy levels to plot
    levels = [10 ** k for k in range(-4, 5)]
    fmt = {value: f'${latex_float(value)}$ keV cm$^2$' for value in levels}
    contours = plt.contour(
        density_interps,
        temperature_interps,
        entropy_interps,
        levels,
        colors='aqua',
        linewidths=0.3
    )

    # work with logarithms for loglog scale
    # middle of the figure:
    # xmin, xmax, ymin, ymax = plt.axis()
    # logmid = (np.log10(xmin) + np.log10(xmax)) / 2, (np.log10(ymin) + np.log10(ymax)) / 2

    label_pos = []
    i = 0
    for line in contours.collections:
        for path in line.get_paths():
            logvert = np.log10(path.vertices)

            # Align with same x-value
            if levels[i] > 1:
                log_rho = -4.5
            else:
                log_rho = 16

            logmid = log_rho, np.log10(levels[i]) - 2 * log_rho / 3
            i += 1

            # find closest point
            logdist = np.linalg.norm(logvert - logmid, ord=2, axis=1)
            min_ind = np.argmin(logdist)
            label_pos.append(10 ** logvert[min_ind, :])

    # Draw contour labels
    plt.clabel(
        contours,
        inline=True,
        inline_spacing=3,
        rightside_up=True,
        colors='aqua',
        fontsize=5,
        fmt=fmt,
        manual=label_pos
    )


def plot_radial_profiles_median(object_database: pd.DataFrame) -> None:
    # Bin objects by mass
    m500crit_log10 = np.array([np.log10(m.value) for m in object_database['M500'].values])
    plot_database = object_database.iloc[np.where(m500crit_log10 > 12)[0]]
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
    assert (x > 0).all(), f"Found negative value(s) in x: {x[x <= 0]}"
    assert (y > 0).all(), f"Found negative value(s) in y: {y[y <= 0]}"

    density_bounds = [1e-6, 1e4]  # in nh/cm^3
    temperature_bounds = [1e3, 1e10]  # in K
    bins = 256

    # Make the norm object to define the image stretch
    density_bins = np.logspace(
        np.log10(density_bounds[0]), np.log10(density_bounds[1]), bins
    )
    temperature_bins = np.logspace(
        np.log10(temperature_bounds[0]), np.log10(temperature_bounds[1]), bins
    )

    fig, (ax0, ax1, ax2) = plt.subplots(
        nrows=3, ncols=1, sharex=True, sharey=True, figsize=(4, 7)
    )
    fig.tight_layout(pad=0.)

    for ax in [ax0, ax1, ax2]:
        ax.loglog()

        # Draw cross-hair marker
        M500 = object_database['M500'].mean()
        R500 = object_database['R500'].mean()
        nH_500 = object_database['nH_500'].mean().value
        T500 = (unyt.G * mean_molecular_weight * M500 * unyt.mass_proton / R500 / 2 / unyt.boltzmann_constant).to(
            'K').value
        ax.hlines(y=T500, xmin=nH_500 / 5, xmax=nH_500 * 5, colors='k', linestyles='-', lw=1)
        ax.vlines(x=nH_500, ymin=T500 / 10, ymax=T500 * 10, colors='k', linestyles='-', lw=1)

        # Star formation threshold
        ax.axvline(0.1, color='k', linestyle=':', lw=1)

    # PLOT ALL PARTICLES ===============================================
    H, density_edges, temperature_edges = np.histogram2d(
        x, y, bins=[density_bins, temperature_bins]
    )

    density_interps, temperature_interps = np.meshgrid(density_bins, temperature_bins)
    temperature_interps *= unyt.K * unyt.boltzmann_constant
    entropy_interps = temperature_interps / (density_interps / unyt.cm ** 3) ** (2 / 3)
    entropy_interps = entropy_interps.to('keV*cm**2').value

    # Define entropy levels to plot
    levels = [10 ** k for k in range(-4, 5)]
    fmt = {value: f'${latex_float(value)}$ keV cm$^2$' for value in levels}
    contours = plt.contour(
        density_interps,
        temperature_interps,
        entropy_interps,
        levels,
        colors='aqua',
        linewidths=0.3
    )

    # work with logarithms for loglog scale
    # middle of the figure:
    # xmin, xmax, ymin, ymax = plt.axis()
    # logmid = (np.log10(xmin) + np.log10(xmax)) / 2, (np.log10(ymin) + np.log10(ymax)) / 2

    label_pos = []
    i = 0
    for line in contours.collections:
        for path in line.get_paths():
            logvert = np.log10(path.vertices)

            # Align with same x-value
            if levels[i] > 1:
                log_rho = -4.5
            else:
                log_rho = 16

            logmid = log_rho, np.log10(levels[i]) - 2 * log_rho / 3
            i += 1

            # find closest point
            logdist = np.linalg.norm(logvert - logmid, ord=2, axis=1)
            min_ind = np.argmin(logdist)
            label_pos.append(10 ** logvert[min_ind, :])

    # Draw contour labels
    plt.clabel(
        contours,
        inline=True,
        inline_spacing=3,
        rightside_up=True,
        colors='aqua',
        fontsize=5,
        fmt=fmt,
        manual=label_pos
    )

    vmax = np.max(H)
    mappable = ax0.pcolormesh(
        density_edges, temperature_edges, H.T,
        norm=LogNorm(vmin=1, vmax=vmax), cmap='Greys_r'
    )
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="3%", pad=0.)
    cb = plt.colorbar(mappable, ax=ax0, cax=cax)
    cb.set_label(label="All particles", size=5)

    # PLOT SN HEATED PARTICLES ===============================================
    H, density_edges, temperature_edges = np.histogram2d(
        x[(snii_flag & ~agn_flag)],
        y[(snii_flag & ~agn_flag)],
        bins=[density_bins, temperature_bins]
    )
    vmax = np.max(H)
    mappable = ax1.pcolormesh(
        density_edges, temperature_edges, H.T,
        norm=LogNorm(vmin=1, vmax=vmax), cmap='Greens_r', alpha=0.6
    )
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="3%", pad=0.)
    cb = plt.colorbar(mappable, ax=ax1, cax=cax)
    cb.set_label(label="SNe heated only")

    # PLOT AGN HEATED PARTICLES ===============================================
    H, density_edges, temperature_edges = np.histogram2d(
        x[(agn_flag & ~snii_flag)],
        y[(agn_flag & ~snii_flag)],
        bins=[density_bins, temperature_bins]
    )
    vmax = np.max(H)
    mappable = ax2.pcolormesh(
        density_edges, temperature_edges, H.T,
        norm=LogNorm(vmin=1, vmax=vmax), cmap='Reds_r', alpha=0.6
    )
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="3%", pad=0.)
    cb = plt.colorbar(mappable, ax=ax2, cax=cax)
    cb.set_label(label="AGN heated only")

    # Heating temperatures
    ax1.axhline(10 ** 7.5, color='k', linestyle='--', lw=1)
    ax2.axhline(10 ** 8.5, color='k', linestyle='--', lw=1)

    ax1.set_ylabel(r"Temperature [K]")
    ax2.set_xlabel(r"Density [$n_H$ cm$^{-3}$]")
    ax0.set_title(
        (
            f"Aperture = {aperture_fraction:.2f} $R_{{500}}$\t\t"
            f"$z = {calibration_zooms.redshift_from_index(args.redshift_index):.2f}$\n"
            f"{''.join(args.keywords)}"
        ),
        fontsize=5
    )

    if not args.quiet:
        plt.show()
    fig.savefig(
        f'{calibration_zooms.output_directory}/density_temperature_{args.redshift_index:04d}.png',
        dpi=300
    )

    plt.close()


if __name__ == "__main__":
    results_database = utils.process_catalogue(_process_single_halo, find_keyword=args.keywords)
    plot_radial_profiles_median(results_database)
