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
M200c = np.asarray([1.487, 3.033, 6.959]) * 1e13
R200c = np.asarray([0.519, 0.658, 0.868])
x = np.asarray([134.688, 90.671, 71.962])
y = np.asarray([169.921, 289.822, 69.291])
z = np.asarray([289.233, 98.227, 240.338])

for i in range(3):
    # EAGLE-XL data path
    dataPath = f"/cosma6/data/dp004/rttw52/EAGLE-XL/EAGLE-XL_ClusterSK{i}_DMO/snapshots/"
    snapFile = dataPath + f"EAGLE-XL_ClusterSK{i}_DMO_0001.hdf5"

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
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_aspect('equal')
    ax.plot(x, y, ',', c="C0", alpha=0.1)
    ax.set_xlim([-size.value, size.value])
    ax.set_ylim([-size.value, size.value])
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
        0 + 1.05 * 5 * R200c[i],
        r"$5 \times R_{200c}$",
        color="grey",
        ha="center",
        va="bottom"
    )
    circle_r200 = plt.Circle((0, 0), R200c[i], color="black", fill=False, linestyle='-')
    circle_5r200 = plt.Circle((0, 0), 5 * R200c[i], color="grey", fill=False, linestyle='--')
    ax.add_artist(circle_r200)
    ax.add_artist(circle_5r200)
    fig.savefig(f"outfiles/halo{i}_particlemap_zoom.png")
    plt.close(fig)
