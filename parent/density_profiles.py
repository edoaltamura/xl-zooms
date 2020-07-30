import numpy as np
import unyt
import matplotlib
matplotlib.use('Agg')
import swiftsimio as sw
# from velociraptor.swift.swift import to_swiftsimio_dataset
# from velociraptor.particles import load_groups
# from velociraptor import load
from matplotlib import pyplot as plt

try:
    plt.style.use("mnras.mplstyle")
except:
    pass

print("Loading halos selected...")
# lines = np.loadtxt("outfiles/halo_selected.txt", comments="#", delimiter=",", unpack=False).T
# print("log10(M200c / Msun): ", np.log10(lines[1] * 1e13))
# print("R200c: ", lines[2])
# print("Centre of potential coordinates: (xC, yC, zC)")
# for i in range(3):
#     print(f"\tHalo {i:d}:\t({lines[3, i]:2.1f}, {lines[4, i]:2.1f}, {lines[5, i]:2.1f})")
# M200c = lines[1] * 1e13
# R200c = lines[2]
# x = lines[3]
# y = lines[4]
# z = lines[5]

print("Loading halos selected...")
M200c = np.asarray([1.487, 3.033, 6.959]) * 1e13
R200c = np.asarray([0.519, 0.658, 0.868])
x = np.asarray([134.688, 90.671, 71.962])
y = np.asarray([169.921, 289.822, 69.291])
z = np.asarray([289.233, 98.227, 240.338])

############################################################
# DENSITY PROFILE FROM SNAPSHOT - ALL PARTICLES
dataPath = "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/"
snapFile = dataPath + "EAGLE-XL_L0300N0564_DMONLY_0036.hdf5"

for i in range(3):
    print(f"Profiling halo {i}...")
    xCen = unyt.unyt_quantity(x[i], unyt.Mpc)
    yCen = unyt.unyt_quantity(y[i], unyt.Mpc)
    zCen = unyt.unyt_quantity(z[i], unyt.Mpc)
    size = unyt.unyt_quantity(6. * R200c[i], unyt.Mpc)
    mask = sw.mask(snapFile)
    region = [
        [xCen - size, xCen + size],
        [yCen - size, yCen + size],
        [zCen - size, zCen + size]
    ]
    mask.constrain_spatial(region)
    data = sw.load(snapFile, mask=mask)
    posDM = data.dark_matter.coordinates / data.metadata.a
    x = posDM[:, 0] - xCen
    y = posDM[:, 1] - yCen
    z = posDM[:, 2] - zCen
    del posDM
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2) / R200c[i]
    masses = np.ones_like(r)

    # constuct bins for the histogram
    lbins = np.logspace(-2, np.log10(5.), 40)
    # compute statistics - each bin has Y value of the sum of the masses of points within the bin X
    hist, bin_edges = np.histogram(r, bins=lbins)
    bin_centre = np.sqrt(bin_edges[1:] * bin_edges[:-1])
    # compute radial density distribution
    volume_shell = (4. * np.pi * (R200c[i] ** 3) / 3.) * ((bin_edges[1:]) ** 3 - (bin_edges[:-1]) ** 3)
    rho_crit = data.metadata.cosmology['Critical density [internal units]'][0]
    densities = hist / volume_shell / rho_crit
    # Plot density profile for each selected halo in volume
    fig, ax = plt.subplots()
    ax.plot(bin_centre, densities, c="C0", linestyle="-")
    ax.set_xlim(1e-2, 6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r"$\rho_{DM}\ /\ \rho_c$")
    ax.set_xlabel(r"$R\ /\ R_{200c}$")
    fig.tight_layout()
    fig.savefig(f"outfiles/halo{i}_density_profile_parent.png")
    plt.close(fig)