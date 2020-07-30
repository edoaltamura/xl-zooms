import numpy as np
import unyt
import matplotlib
matplotlib.use('Agg')
import swiftsimio as sw
import matplotlib.pyplot as plt

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


print("Loading halos selected...")
lines = np.loadtxt("outfiles/halo_selected_SK.txt", comments="#", delimiter=",", unpack=False).T
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

# EAGLE-XL data path
dataPath = "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/"
snapFile = dataPath + "EAGLE-XL_L0300N0564_DMONLY_0036.hdf5"

for i in range(3):
    print(f"Rendering halo {i}...")
    # Load data using mask
    xCen = unyt.unyt_quantity(x[i], unyt.Mpc)
    yCen = unyt.unyt_quantity(y[i], unyt.Mpc)
    zCen = unyt.unyt_quantity(z[i], unyt.Mpc)
    size = unyt.unyt_quantity(5.5 * R200c[i], unyt.Mpc)
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
    x = posDM[:, 0] - xCen
    y = posDM[:, 1] - yCen
    z = posDM[:, 2] - zCen
    del posDM

    # Make figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=1024 // 8)
    ax.set_aspect('equal')
    ax.plot(x, y, ',', c="C0", alpha=1)
    ax.set_xlim([-size.value, size.value])
    ax.set_ylim([-size.value, size.value])
    ax.set_ylabel(r"$x$ [Mpc]")
    ax.set_xlabel(r"$y$ [Mpc]")
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
    fig.savefig(f"outfiles/halo{i}_particlemap_parent.png")
    plt.close(fig)
