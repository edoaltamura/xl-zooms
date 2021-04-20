from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.legend_handler import HandlerLine2D
import unyt
import numpy as np
import swiftsimio as sw
import velociraptor as vr
from typing import Tuple
import matplotlib.patheffects as path_effects

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

import sys

# Make the register backend visible to the script
sys.path.append("..")
from literature import Sun2009, Cosmology

Sun2009 = Sun2009()
fb = Cosmology().fb

# Constants
mean_molecular_weight = 0.59
mean_atomic_weight_per_free_electron = 1.14


def update_prop(handle, orig):
    handle.update_from(orig)
    handle.set_marker("o")


def histogram_unyt(
        data: unyt.unyt_array,
        bins: unyt.unyt_array = None,
        weights: unyt.unyt_array = None,
        **kwargs,
) -> Tuple[unyt.unyt_array]:
    assert data.shape == weights.shape, (
        "Data and weights arrays must have the same shape. "
        f"Detected data {data.shape}, weights {weights.shape}."
    )

    assert data.units == bins.units, (
        "Data and bins must have the same units. "
        f"Detected data {data.units}, bins {bins.units}."
    )

    hist, bin_edges = np.histogram(data.value, bins=bins.value, weights=weights.value, **kwargs)
    hist *= weights.units
    bin_edges *= data.units

    return hist, bin_edges


def profile_3d_particles(
        path_to_snap: str,
        path_to_catalogue: str,
) -> tuple:
    # Read in halo properties
    vr_catalogue_handle = vr.load(path_to_catalogue)
    M500 = vr_catalogue_handle.spherical_overdensities.mass_500_rhocrit[0].to('Msun')
    R500 = vr_catalogue_handle.spherical_overdensities.r_500_rhocrit[0].to('Mpc')
    XPotMin = vr_catalogue_handle.positions.xcminpot[0].to('Mpc')
    YPotMin = vr_catalogue_handle.positions.ycminpot[0].to('Mpc')
    ZPotMin = vr_catalogue_handle.positions.zcminpot[0].to('Mpc')

    # Apply spatial mask to particles. SWIFTsimIO needs comoving coordinates
    # to filter particle coordinates, while VR outputs are in physical units.
    # Convert the region bounds to comoving, but keep the CoP and Rcrit in
    # physical units for later use.
    mask = sw.mask(path_to_snap, spatial_only=True)
    region = [
        [(XPotMin - R500), (XPotMin + R500)],
        [(YPotMin - R500), (YPotMin + R500)],
        [(ZPotMin - R500), (ZPotMin + R500)]
    ]
    mask.constrain_spatial(region)
    data = sw.load(path_to_snap, mask=mask)

    # Select hot gas within sphere
    tempGas = data.gas.temperatures
    deltaX = data.gas.coordinates[:, 0] - XPotMin
    deltaY = data.gas.coordinates[:, 1] - YPotMin
    deltaZ = data.gas.coordinates[:, 2] - ZPotMin
    radial_distance = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) / R500
    index = np.where((radial_distance < 2) & (tempGas > 1e5))[0]
    del tempGas, deltaX, deltaY, deltaZ

    mass_weighted_temperatures = (data.gas.temperatures * unyt.boltzmann_constant).to('keV')
    number_densities = (data.gas.densities.to('g/cm**3') / (unyt.mp * mean_molecular_weight)).to('cm**-3')
    field_value = mass_weighted_temperatures / number_densities ** (2 / 3)

    radial_distance = radial_distance[index]
    entropies = field_value[index]
    temperatures = mass_weighted_temperatures[index]

    rho_crit = unyt.unyt_quantity(
        data.metadata.cosmology.critical_density(data.metadata.z).value, 'g/cm**3'
    ).to('Msun/Mpc**3')
    densities = data.gas.densities[index] / rho_crit

    return radial_distance, densities, temperatures, entropies, M500, R500


def profile_3d_shells(
        path_to_snap: str,
        path_to_catalogue: str,
) -> tuple:
    # Read in halo properties
    vr_catalogue_handle = vr.load(path_to_catalogue)
    M500 = vr_catalogue_handle.spherical_overdensities.mass_500_rhocrit[0].to('Msun')
    R500 = vr_catalogue_handle.spherical_overdensities.r_500_rhocrit[0].to('Mpc')
    XPotMin = vr_catalogue_handle.positions.xcminpot[0].to('Mpc')
    YPotMin = vr_catalogue_handle.positions.ycminpot[0].to('Mpc')
    ZPotMin = vr_catalogue_handle.positions.zcminpot[0].to('Mpc')

    # Apply spatial mask to particles. SWIFTsimIO needs comoving coordinates
    # to filter particle coordinates, while VR outputs are in physical units.
    # Convert the region bounds to comoving, but keep the CoP and Rcrit in
    # physical units for later use.
    mask = sw.mask(path_to_snap, spatial_only=True)
    region = [
        [(XPotMin - R500), (XPotMin + R500)],
        [(YPotMin - R500), (YPotMin + R500)],
        [(ZPotMin - R500), (ZPotMin + R500)]
    ]
    mask.constrain_spatial(region)
    data = sw.load(path_to_snap, mask=mask)

    # Select gas within sphere and main FOF halo
    fof_id = data.gas.fofgroup_ids
    tempGas = data.gas.temperatures
    deltaX = data.gas.coordinates[:, 0] - XPotMin
    deltaY = data.gas.coordinates[:, 1] - YPotMin
    deltaZ = data.gas.coordinates[:, 2] - ZPotMin
    radial_distance = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) / R500
    index = np.where((radial_distance < 2) & (fof_id == 1) & (tempGas > 1e5))[0]
    del deltaX, deltaY, deltaZ, fof_id, tempGas

    radial_distance = radial_distance[index]
    data.gas.masses = data.gas.masses[index]
    data.gas.temperatures = data.gas.temperatures[index]

    # Define radial bins and shell volumes
    lbins = np.logspace(-3, 2, 40) * radial_distance.units
    radial_bin_centres = 10.0 ** (0.5 * np.log10(lbins[1:] * lbins[:-1])) * radial_distance.units
    volume_shell = (4. * np.pi / 3.) * (R500 ** 3) * ((lbins[1:]) ** 3 - (lbins[:-1]) ** 3)

    mass_weights, _ = histogram_unyt(radial_distance, bins=lbins, weights=data.gas.masses)
    mass_weights[mass_weights == 0] = np.nan  # Replace zeros with Nans
    density_profile = mass_weights / volume_shell
    number_density_profile = (density_profile.to('g/cm**3') / (unyt.mp * mean_molecular_weight)).to('cm**-3')

    mass_weighted_temperatures = (data.gas.temperatures * unyt.boltzmann_constant).to('keV') * data.gas.masses
    temperature_weights, _ = histogram_unyt(radial_distance, bins=lbins, weights=mass_weighted_temperatures)
    temperature_weights[temperature_weights == 0] = np.nan  # Replace zeros with Nans
    temperature_profile = temperature_weights / mass_weights  # kBT in units of [keV]

    entropy_profile = temperature_profile / number_density_profile ** (2 / 3)

    rho_crit = unyt.unyt_quantity(
        data.metadata.cosmology.critical_density(data.metadata.z).value, 'g/cm**3'
    ).to('Msun/Mpc**3')
    density_profile /= rho_crit

    return radial_bin_centres, density_profile, temperature_profile, entropy_profile, M500, R500


cwd = '/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/'
runname = 'L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1'

paths = {
    '0.': (
        cwd + 'vr_partial_outputs/alpha0p0.properties',
        cwd + runname + '_alpha0p0/snapshots/' + runname + '_SNnobirth_2252.hdf5'
    ),
    '0.5': (
        cwd + 'vr_partial_outputs/alpha0p5.properties',
        cwd + runname + '_alpha0p5/snapshots/' + runname + '_SNnobirth_2252.hdf5'
    ),
    '0.7': (
        cwd + 'vr_partial_outputs/alpha0p7.properties',
        cwd + runname + '_alpha0p7/snapshots/' + runname + '_SNnobirth_2252.hdf5'
    ),
    '0.9': (
        cwd + 'vr_partial_outputs/alpha0p9.properties',
        cwd + runname + '_alpha0p9/snapshots/' + runname + '_SNnobirth_2252.hdf5'
    ),
    '1.': (
        cwd + 'vr_partial_outputs/alpha1p0.properties',
        cwd + runname + '_alpha1p0/snapshots/' + runname + '_SNnobirth_2252.hdf5'
    ),
    'adi0.': (
        cwd + 'vr_partial_outputs/alpha0p0_adi.properties',
        cwd + runname + '_alpha1p0_adiabatic/snapshots/' + runname + '_SNnobirth_2252.hdf5'
    ),
    'adi1.': (
        cwd + 'vr_partial_outputs/alpha1p0_adi.properties',
        cwd + runname + '_alpha1p0_adiabatic/snapshots/' + runname + '_SNnobirth_2252.hdf5'
    ),
}

alpha_list = ['adi1.', '1.']

name = "viridis"
cmap = get_cmap(name)

fig = plt.figure(figsize=(9, 5))
gs = fig.add_gridspec(2, 3, hspace=0.05, wspace=0.3)
axes = gs.subplots(sharex=True, sharey=False)

shadow = dict(path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])

for ax in axes.flat:
    ax.loglog()
    ax.axvline(x=1, color='k', linestyle='--')
    ax.set_xlim([1e-3, 2])

print("Entropy - shell average")
ax = axes[0, 0]
for alpha_key in alpha_list:
    print(alpha_key)
    cat, snap = paths[alpha_key]

    radial_bin_centres, _, _, entropy_profile, M500, R500 = profile_3d_shells(
        path_to_snap=snap,
        path_to_catalogue=cat,
    )

    ax.plot(
        radial_bin_centres,
        entropy_profile,
        linestyle='-',
        color=cmap(float(alpha_key)),
        linewidth=1,
        alpha=1,
    )

    kBT500 = (
            unyt.G * mean_molecular_weight * M500 * unyt.mass_proton / R500 / 2
    ).to('keV')
    K500 = (
            kBT500 / (3 * M500 * fb / (4 * np.pi * R500 ** 3 * unyt.mass_proton)) ** (2 / 3)
    ).to('keV*cm**2')

    ax.axhline(y=K500, color=cmap(float(alpha_key)), linestyle='--', **shadow)
    ax.set_ylabel(r'$K$ [keV cm$^2$]')
    ax.set_ylim([30, 1e4])
ax.text(
    ax.get_xlim()[0], K500, r'$K_{500}$',
    horizontalalignment='left',
    verticalalignment='bottom',
    color='k',
    bbox=dict(
        boxstyle='square,pad=10',
        fc='none',
        ec='none'
    )
)
Sun2009.overlay_entropy_profiles(axes=ax, k_units='keVcm^2')

print("Entropy - dot particles")
ax = axes[1, 0]
for alpha_key in alpha_list:
    print(alpha_key)
    cat, snap = paths[alpha_key]

    radial_distance, _, _, entropies, M500, R500 = profile_3d_particles(
        path_to_snap=snap,
        path_to_catalogue=cat,
    )

    ax.plot(radial_distance[::10], entropies[::10], marker=',', lw=0, linestyle="", c=cmap(float(alpha_key)),
            alpha=0.5, label=f"Alpha_max = {alpha_key}")

    kBT500 = (
            unyt.G * mean_molecular_weight * M500 * unyt.mass_proton / R500 / 2
    ).to('keV')
    K500 = (
            kBT500 / (3 * M500 * fb / (4 * np.pi * R500 ** 3 * unyt.mass_proton)) ** (2 / 3)
    ).to('keV*cm**2')

    ax.axhline(y=K500, color=cmap(float(alpha_key)), linestyle='--', **shadow)
    ax.set_ylabel(r'$K$ [keV cm$^2$]')
    ax.set_ylim([30, 1e4])
ax.text(
    ax.get_xlim()[0], K500, r'$K_{500}$',
    horizontalalignment='left',
    verticalalignment='bottom',
    color='k',
    bbox=dict(
        boxstyle='square,pad=10',
        fc='none',
        ec='none'
    )
)
Sun2009.overlay_entropy_profiles(axes=ax, k_units='keVcm^2')

print("Temperatures - shell average")
ax = axes[0, 1]
for alpha_key in alpha_list:
    print(alpha_key)
    cat, snap = paths[alpha_key]

    radial_bin_centres, _, temperature_profile, _, M500, R500 = profile_3d_shells(
        path_to_snap=snap,
        path_to_catalogue=cat,
    )

    ax.plot(
        radial_bin_centres,
        temperature_profile,
        linestyle='-',
        color=cmap(float(alpha_key)),
        linewidth=1,
        alpha=1,
    )

    kBT500 = (
            unyt.G * mean_molecular_weight * M500 * unyt.mass_proton / R500 / 2
    ).to('keV')

    ax.axhline(y=kBT500, color=cmap(float(alpha_key)), linestyle='--', **shadow)
    ax.set_ylabel(r'$k_BT$ [keV]')
    ax.set_ylim([1, 6])
ax.text(
    ax.get_xlim()[0], kBT500, r'$k_BT_{500}$',
    horizontalalignment='left',
    verticalalignment='bottom',
    color='k',
    bbox=dict(
        boxstyle='square,pad=10',
        fc='none',
        ec='none'
    )
)

print("Temperatures - dot particles")
ax = axes[1, 1]
for alpha_key in alpha_list:
    print(alpha_key)
    cat, snap = paths[alpha_key]

    radial_distance, _, temperatures, _, M500, R500 = profile_3d_particles(
        path_to_snap=snap,
        path_to_catalogue=cat,
    )

    ax.plot(radial_distance[::10], temperatures[::10], marker=',', lw=0, linestyle="", c=cmap(float(alpha_key)),
            alpha=0.5, label=f"Alpha_max = {alpha_key}")

    kBT500 = (
            unyt.G * mean_molecular_weight * M500 * unyt.mass_proton / R500 / 2
    ).to('keV')

    ax.axhline(y=kBT500, color=cmap(float(alpha_key)), linestyle='--', **shadow)
    ax.set_ylabel(r'$k_BT$ [keV]')
    ax.set_ylim([0.1, 10])
    ax.set_xlabel(f'$r/r_{{500,true}}$')
ax.text(
    ax.get_xlim()[0], kBT500, r'$k_BT_{500}$',
    horizontalalignment='left',
    verticalalignment='bottom',
    color='k',
    bbox=dict(
        boxstyle='square,pad=10',
        fc='none',
        ec='none'
    )
)

print("Density - shell average")
ax = axes[0, 2]
for alpha_key in alpha_list:
    print(alpha_key)
    cat, snap = paths[alpha_key]

    radial_bin_centres, density_profile, _, _, M500, R500 = profile_3d_shells(
        path_to_snap=snap,
        path_to_catalogue=cat,
    )

    ax.plot(
        radial_bin_centres,
        density_profile,
        linestyle='-',
        color=cmap(float(alpha_key)),
        linewidth=1,
        alpha=1,
    )

    ax.set_ylabel(r'$\rho_{gas} / \rho_{crit}$')
    ax.set_ylim([1, 1e3])

print("Density - dot particles")
ax = axes[1, 2]
for alpha_key in alpha_list:
    print(alpha_key)
    cat, snap = paths[alpha_key]

    radial_distance, densities, _, _, M500, R500 = profile_3d_particles(
        path_to_snap=snap,
        path_to_catalogue=cat,
    )

    ax.plot(
        radial_distance[::10],
        densities[::10],
        marker=',',
        lw=0, linestyle="", c=cmap(float(alpha_key)),
        alpha=0.5, label=f"Alpha_max = {alpha_key}"
    )

    ax.set_ylabel(r'$\rho_{gas} / \rho_{crit}$')
    ax.set_ylim([1, 1e3])

plt.legend(handler_map={plt.Line2D: HandlerLine2D(update_func=update_prop)})

fig.suptitle(
    (
        f"Aperture = 2 $r_{{500}}$\t\t"
        f"$z = 0$\n"
        f"VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_alpha*_no_adiabatic\n"
        f"Central FoF group only\n"
        f"Top row: shell-averaged, bottom row: particle-dots ([::20])"
    ),
    fontsize=7
)

plt.show()
