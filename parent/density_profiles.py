import matplotlib
matplotlib.use('Agg')

import numpy as np
import unyt
import swiftsimio as sw
from matplotlib import pyplot as plt

try:
    plt.style.use("mnras.mplstyle")
except:
    pass

##########################################################
# INPUTS

author = "SK"
out_to_radius = 5

dataPath = "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/"
snapFile = dataPath + "EAGLE-XL_L0300N0564_DMONLY_0036.hdf5"

##########################################################

print("Loading halos selected...")
lines = np.loadtxt(f"outfiles/halo_selected_{author}.txt", comments="#", delimiter=",", unpack=False).T
print("log10(M200c / Msun): ", np.log10(lines[1] * 1e13))
print("R200c: ", lines[2])
print("Centre of potential coordinates: (xC, yC, zC)")
for i in range(3):
    print(f"\tHalo {i:d}:\t({lines[3, i]:2.1f}, {lines[4, i]:2.1f}, {lines[5, i]:2.1f})")
M200c = lines[1] * 1e13
R200c = lines[2]
x = lines[3]
y = lines[4]
z = lines[5]

############################################################
# DENSITY PROFILE FROM SNAPSHOT - ALL PARTICLES

def density_profile(halo_id: int, outfig: bool = False):

    print(f"Profiling halo {halo_id}...")
    xCen = unyt.unyt_quantity(x[halo_id], unyt.Mpc)
    yCen = unyt.unyt_quantity(y[halo_id], unyt.Mpc)
    zCen = unyt.unyt_quantity(z[halo_id], unyt.Mpc)
    size = unyt.unyt_quantity(out_to_radius * R200c[halo_id], unyt.Mpc)
    mask = sw.mask(snapFile)
    region = [
        [xCen - size, xCen + size],
        [yCen - size, yCen + size],
        [zCen - size, zCen + size]
    ]
    mask.constrain_spatial(region)
    data = sw.load(snapFile, mask=mask)
    posDM = data.dark_matter.coordinates / data.metadata.a
    r = np.sqrt(
        (posDM[:, 0] - xCen) ** 2 +
        (posDM[:, 1] - yCen) ** 2 +
        (posDM[:, 2] - zCen) ** 2
    ) / R200c[halo_id]

    # Calculate particle mass and rho_crit
    unitLength = data.metadata.units.length
    unitMass = data.metadata.units.mass
    rho_crit = unyt.unyt_quantity(
        data.metadata.cosmology['Critical density [internal units]'],
        unitMass / unitLength ** 3
    )
    rhoMean = rho_crit * data.metadata.cosmology['Omega_m']
    vol = data.metadata.boxsize[0] ** 3
    numPart = data.metadata.n_dark_matter
    particleMass = rhoMean * vol / numPart

    # constuct bins for the histogram
    lbins = np.logspace(-2, np.log10(out_to_radius), 40)
    # compute statistics - each bin has Y value of the sum of the masses of points within the bin X
    hist, bin_edges = np.histogram(r, bins=lbins)
    bin_centre = np.sqrt(bin_edges[1:] * bin_edges[:-1])
    volume_shell = (4. * np.pi / 3.) * (R200c[halo_id] ** 3) * ((bin_edges[1:]) ** 3 - (bin_edges[:-1]) ** 3)
    densities = hist / volume_shell / rho_crit

    # Plot density profile for each selected halo in volume
    fig, ax = plt.subplots()
    ax.plot(bin_centre, densities, c="C0", linestyle="-")
    ax.set_xlim(1e-2, out_to_radius)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r"$\rho_{DM}\ /\ \rho_c$")
    ax.set_xlabel(r"$R\ /\ R_{200c}$")
    fig.tight_layout()
    fig.savefig(f"outfiles/halo{halo_id}{author}_density_profile_parent.png")
    if outfig:
        return fig, ax
    else:
        plt.close(fig)
        return

for i in range(3):
    density_profile(i)