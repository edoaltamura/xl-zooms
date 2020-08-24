# DENSITY PROFILE FROM SNAPSHOT - ALL PARTICLES

import matplotlib

matplotlib.use('Agg')

import numpy as np
import unyt
import h5py
import swiftsimio as sw
from matplotlib import pyplot as plt

try:
    plt.style.use("mnras.mplstyle")
except:
    pass


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


# INPUTS
author = "SK"
halo_id = 0
out_to_radius = 3

snap_filepath = f"/cosma6/data/dp004/rttw52/EAGLE-XL/EAGLE-XL_ClusterSK{halo_id}_DMO/snapshots/EAGLE-XL_ClusterSK{halo_id}_DMO_0001.hdf5"
velociraptor_properties = f"/cosma6/data/dp004/dc-alta2/xl-zooms/halo_{author}{halo_id}_0001/halo_{author}{halo_id}_0001.properties.0"
output_directory = "outfiles/"

#############################################

# Load velociraptor data
with h5py.File(velociraptor_properties, 'r') as vr_file:
    M200c = vr_file['/Mass_200crit'][0] * 1e10
    R200c = vr_file['/R_200crit'][0]
    Xcminpot = vr_file['/Xcminpot'][0]
    Ycminpot = vr_file['/Ycminpot'][0]
    Zcminpot = vr_file['/Zcminpot'][0]

M200c = unyt.unyt_quantity(M200c, unyt.Solar_Mass )
R200c = unyt.unyt_quantity(R200c, unyt.Mpc)
xCen = unyt.unyt_quantity(Xcminpot, unyt.Mpc)
yCen = unyt.unyt_quantity(Ycminpot, unyt.Mpc)
zCen = unyt.unyt_quantity(Zcminpot, unyt.Mpc)

# Construct spatial mask to feed into swiftsimio
size = out_to_radius * R200c
mask = sw.mask(snap_filepath)
region = [
    [xCen - size, xCen + size],
    [yCen - size, yCen + size],
    [zCen - size, zCen + size]
]
mask.constrain_spatial(region)
data = sw.load(snap_filepath, mask=mask)

# Get DM particle coordinates and compute radial distance from CoP in R200 units
posDM = data.dark_matter.coordinates / data.metadata.a
r = np.sqrt(
    (posDM[:, 0] - xCen) ** 2 +
    (posDM[:, 1] - yCen) ** 2 +
    (posDM[:, 2] - zCen) ** 2
) / R200c

# Calculate particle mass and rho_crit
unitLength = data.metadata.units.length
unitMass = data.metadata.units.mass
rho_crit = unyt.unyt_quantity(
    data.metadata.cosmology['Critical density [internal units]'],
    unitMass / unitLength ** 3
)
particleMasses = data.dark_matter.masses.to('Msun')
zoom_mass_resolution = particleMasses

# Construct bins and compute density profile
lbins = np.logspace(-2, np.log10(out_to_radius), 40)
hist, bin_edges = np.histogram(r, bins=lbins, weights=particleMasses)
bin_centre = np.sqrt(bin_edges[1:] * bin_edges[:-1])
volume_shell = (4. * np.pi / 3.) * (R200c ** 3) * ((bin_edges[1:]) ** 3 - (bin_edges[:-1]) ** 3)
densities = hist / volume_shell / rho_crit

# Plot density profile
fig, ax = plt.subplots()
zoom_label = f'Zoom: $m_\\mathrm{{DM}} = {latex_float(zoom_mass_resolution.value[0])}\\ {zoom_mass_resolution.units.latex_repr}$'
ax.plot(bin_centre, densities, c="orange", linestyle="-", label=zoom_label)

ax.text(
    0.025,
    0.025,
    (
        f"Halo {halo_id:d} DMO\n"
        f"$z={data.metadata.z:3.3f}$\n"
        "Zoom VR output:\n"
        f"$M_{{200c}}={latex_float(M200c.value)}\\ {M200c.units.latex_repr}$\n"
        f"$R_{{200c}}={latex_float(R200c.value)}\\ {R200c.units.latex_repr}$"
    ),
    color="black",
    ha="left",
    va="bottom",
    transform=ax.transAxes,
)

ax.set_xlim(1e-2, out_to_radius)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel(r"$\rho_{DM}\ /\ \rho_c$")
ax.set_xlabel(r"$R\ /\ R_{200c}$")
plt.legend()
fig.tight_layout()
fig.savefig(f"{output_directory}halo{halo_id}{author}_density_profile_zoom.png")
plt.close(fig)
