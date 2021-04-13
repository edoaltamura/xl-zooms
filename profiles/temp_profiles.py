from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
import unyt
import numpy as np
import swiftsimio as sw
import velociraptor as vr

# Constants
mean_molecular_weight = 0.59
mean_atomic_weight_per_free_electron = 1.14


def profile_3d_single_halo(
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

    field_label = r'$K$ [keV cm$^2$]'
    radial_distance = radial_distance[index]
    field_value = field_value[index]
    field_masses = data.gas.temperatures[index]

    return radial_distance, field_value, field_masses, field_label, M500, R500


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
}

name = "Spectral"
cmap = get_cmap(name)

fig, ax = plt.subplots()
ax.loglog()

for alpha_key in paths:
    print(alpha_key)
    cat, snap = paths[alpha_key]

    output = profile_3d_single_halo(
        path_to_snap=snap,
        path_to_catalogue=cat,
    )
    print(output)
    radial_distance, field_value, field_masses, field_label, M500, R500 = output

    ax.plot(radial_distance[::20], field_value[::20], marker=',', lw=0, linestyle="", c=cmap(float(alpha_key)),
            alpha=0.1, label=alpha_key)

ax.set_xlabel(f'$r/r_{{500,true}}$')
ax.set_ylabel(field_label)
ax.set_xlim([0.05, 2])
ax.set_ylim([30, 1e4])

from matplotlib.legend_handler import HandlerLine2D


def update_prop(handle, orig):
    handle.update_from(orig)
    handle.set_marker("o")


plt.legend(handler_map={plt.Line2D: HandlerLine2D(update_func=update_prop)})
plt.show()
