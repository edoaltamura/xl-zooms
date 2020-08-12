import numpy as np
import unyt
import h5py
import matplotlib
matplotlib.use('Agg')
import swiftsimio as sw
from matplotlib import pyplot as plt

try:
    plt.style.use("mnras.mplstyle")
except:
    pass

# INPUTS
author = "SK"
metadata_filepath = "outfiles/halo_selected_SK.txt"
simdata_dirpath = "/cosma6/data/dp004/rttw52/EAGLE-XL/"
snap_relative_filepaths = [
    f"EAGLE-XL_ClusterSK{i}_DMO/snapshots/EAGLE-XL_ClusterSK{i}_DMO_0001.hdf5"
    for i in range(3)
]

velociraptor_properties = [
    f"/cosma6/data/dp004/dc-alta2/xl-zooms/halo_{author}{i}_0001/halo_{author}{i}_0001.properties.0"
    for i in range(3)
]

output_directory = "outfiles/"


#############################################

print("Loading halos selected...")
# lines = np.loadtxt(f"outfiles/halo_selected_{author}.txt", comments="#", delimiter=",", unpack=False).T
# print("log10(M200c / Msun): ", np.log10(lines[1] * 1e13))
# print("R200c: ", lines[2])
# print("Centre of potential coordinates: (xC, yC, zC)")
# for i in range(3):
#     print(f"\tHalo {i:d}:\t({lines[3, i]:2.1f}, {lines[4, i]:2.1f}, {lines[5, i]:2.1f})")
M200c = []
R200c = []
x = []
y = []
z = []

for vr_path in velociraptor_properties:
    with h5py.File(vr_path, 'r') as vr_file:
        M200c.append(vr_file['/Mass_200crit'][0] * 1e10)
        R200c.append(vr_file['/R_200crit'][0])
        x.append(vr_file['/Xcminpot'][0])
        y.append(vr_file['/Ycminpot'][0])
        z.append(vr_file['/Zcminpot'][0])

############################################################
# DENSITY PROFILE FROM SNAPSHOT - ALL PARTICLES

for i in range(len(snap_relative_filepaths)):
    # EAGLE-XL data path
    snapFile = simdata_dirpath + snap_relative_filepaths[i]
    print(f"Profiling {snap_relative_filepaths[i]}...")
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
    r = np.sqrt(
        (posDM[:, 0] - xCen) ** 2 +
        (posDM[:, 1] - yCen) ** 2 +
        (posDM[:, 2] - zCen) ** 2
    ) / R200c[i]
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
    fig.savefig(f"{output_directory}halo{i}{author}_density_profile_zoom.png")
    plt.close(fig)