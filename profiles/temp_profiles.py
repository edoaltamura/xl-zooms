from os.path import isfile
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
    a = vr_catalogue_handle.a
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
        cwd + runname + '_alpha0p0/snapshots/' + runname + '_2252.hdf5'
    ),
    '0.5': (
        cwd + 'vr_partial_outputs/alpha0p5.properties',
        cwd + runname + '_alpha0p5/snapshots/' + runname + '_2252.hdf5'
    ),
    '0.7': (
        cwd + 'vr_partial_outputs/alpha0p7.properties',
        cwd + runname + '_alpha0p7/snapshots/' + runname + '_2252.hdf5'
    ),
    '0.9': (
        cwd + 'vr_partial_outputs/alpha0p9.properties',
        cwd + runname + '_alpha0p9/snapshots/' + runname + '_2252.hdf5'
    ),
    '1.': (
        cwd + 'vr_partial_outputs/alpha1p0.properties',
        cwd + runname + '_alpha1p0/snapshots/' + runname + '_2252.hdf5'
    ),
}

name = "Spectral"
cmap = get_cmap(name)

fig, ax = plt.subplots()

for alpha_key in paths:
    print(alpha_key)
    cat, snap = paths[alpha_key]

    output = profile_3d_single_halo(
        path_to_snap=snap,
        path_to_catalogue=cat,
    )
    radial_distance, field_value, field_masses, field_label, M500, R500 = output

    ax.plot(radial_distance[::20], field_value[::20], marker=',', lw=0, linestyle="", c=cmap(float(alpha_key)),
            alpha=0.1)

plt.show()
