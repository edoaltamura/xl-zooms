import matplotlib
matplotlib.use('Agg')

import numpy as np
import unyt
import h5py
import matplotlib.pyplot as plt
import swiftsimio as sw


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

#############################################
# INPUTS
author = "SK"
out_to_radius = 5

metadata_filepath = f"outfiles/halo_selected_{author}.txt"
simdata_dirpath = "/cosma6/data/dp004/rttw52/EAGLE-XL/"

snap_relative_filepaths = ['EAGLE-XL_ClusterSK0_High/snapshots/EAGLE-XL_ClusterSK0_High_0001.hdf5']
velociraptor_properties = ["/cosma6/data/dp004/dc-alta2/xl-zooms/halo_SK0_High_0001/halo_SK0_High_0001.properties.0"]
output_directory = "outfiles/"


#############################################

print("Loading halos selected...")
# lines = np.loadtxt(f"outfiles/halo_selected_{author}.txt", comments="#", delimiter=",", unpack=False).T
# print("log10(M200c / Msun): ", np.log10(lines[1] * 1e13))
# print("r200c: ", lines[2])
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


for i in range(len(snap_relative_filepaths)):
    # EAGLE-XL data path
    snapFile = simdata_dirpath + snap_relative_filepaths[i]
    print(f"Rendering {snap_relative_filepaths[i]}...")
    # Load data using mask
    xCen = unyt.unyt_quantity(x[i], unyt.Mpc)
    yCen = unyt.unyt_quantity(y[i], unyt.Mpc)
    zCen = unyt.unyt_quantity(z[i], unyt.Mpc)
    size = unyt.unyt_quantity(out_to_radius * R200c[i], unyt.Mpc)
    mask = sw.mask(snapFile)
    region = [
        [xCen - size, xCen + size],
        [yCen - size, yCen + size],
        [zCen - size, zCen + size]
    ]
    mask.constrain_spatial(region)

    # Load data using mask
    data = sw.load(snapFile, mask=mask)
    posDM = data.dark_matter.coordinates / data.metadata.a
    coord_x = posDM[:, 0] - xCen
    coord_y = posDM[:, 1] - yCen
    coord_z = posDM[:, 2] - zCen
    del posDM

    # Make figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=1024 // 8)
    ax.set_aspect('equal')
    ax.plot(coord_x, coord_y, ',', c="C0", alpha=0.1)
    ax.set_xlim([-size.value, size.value])
    ax.set_ylim([-size.value, size.value])
    ax.set_ylabel(r"$y$ [Mpc]")
    ax.set_xlabel(r"$x$ [Mpc]")
    ax.text(
        0.025,
        0.975,
        f"Halo {i:d} DMO\n",
        color="black",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.975,
        0.975,
        f"$z={data.metadata.z:3.3f}$",
        color="black",
        ha="right",
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.975,
        0.025,
        (
            f"$M_{{200c}}={latex_float(M200c[i])}$ M$_\odot$"
        ),
        color="black",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )
    ax.text(
        0,
        0 + 1.05 * R200c[i],
        r"$R_{200c}$",
        color="black",
        ha="center",
        va="bottom"
    )
    ax.text(
        0,
        0 + 1.002 * 5 * R200c[i],
        r"$5 \times R_{200c}$",
        color="grey",
        ha="center",
        va="bottom"
    )
    circle_r200 = plt.Circle((0, 0), R200c[i], color="black", fill=False, linestyle='-')
    circle_5r200 = plt.Circle((0, 0), 5 * R200c[i], color="grey", fill=False, linestyle='--')
    ax.add_artist(circle_r200)
    ax.add_artist(circle_5r200)
    fig.savefig(f"{output_directory}halo{i}{author}_particlemap{out_to_radius}r200_zoom.png")
    plt.close(fig)
