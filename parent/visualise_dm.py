import numpy as np
import unyt
import matplotlib
matplotlib.use('Agg')
import swiftsimio as sw
from swiftsimio.visualisation.projection import project_pixel_grid
from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def dm_render(swio_data, region: list = None, resolution: int = 1024):
    # Generate smoothing lengths for the dark matter
    swio_data.dark_matter.smoothing_lengths = generate_smoothing_lengths(
        swio_data.dark_matter.coordinates,
        swio_data.metadata.boxsize,
        kernel_gamma=1.8,
        neighbours=57,
        speedup_fac=2,
        dimension=3,
    )
    # Project the dark matter mass
    dm_map = project_pixel_grid(
        # Note here that we pass in the dark matter dataset not the whole
        # data object, to specify what particle type we wish to visualise
        data=swio_data.dark_matter,
        boxsize=swio_data.metadata.boxsize,
        resolution=resolution,
        project=None,
        parallel=True,
        region=region
    )
    return dm_map

author = "SK"
out_to_radius = 5

print("Loading halos selected...")
lines = np.loadtxt(f"outfiles/halo_selected_{author}.txt", comments="#", delimiter=",", unpack=False).T
print("log10(m200c / Msun): ", np.log10(lines[1] * 1e13))
print("r200c: ", lines[2])
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

render_volume = False
if render_volume:
    print("\nRendering snapshot volume...")
    data = sw.load(snapFile)
    dm_mass = dm_render(data, resolution=1024)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=1024 // 8)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    ax.imshow(dm_mass, norm=LogNorm(), cmap="inferno", origin="lower")
    ax.text(
        0.975,
        0.975,
        f"$z={data.metadata.z:3.3f}$",
        color="white",
        ha="right",
        va="top",
        transform=ax.transAxes,
    )
    for i in range(3):
        circle_10r200 = plt.Circle((x[i], y[i]), 10 * R200c[i], color="white", fill=False, linestyle='-')
        ax.add_artist(circle_10r200)
        ax.text(
            x[i],
            y[i] + 1.05 * 10 * R200c[i],
            f"{i}",
            color="white",
            ha="center",
            va="bottom"
        )
    fig.savefig(f"outfiles/volume_DMmap.png")
    plt.close(fig)

for i in range(3):
    print(f"Rendering halo {i}...")
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
    dm_mass = dm_render(data, region=(region[0] + region[1]))

    # Make figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=1024 // 8)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    ax.imshow(dm_mass.T, norm=LogNorm(), cmap="inferno", origin="lower", extent=(region[0] + region[1]))
    ax.text(
        0.025,
        0.975,
        f"Halo {i:d} DMO\n",
        color="white",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.975,
        0.975,
        f"$z={data.metadata.z:3.3f}$",
        color="white",
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
        color="white",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )
    ax.text(
        xCen.value,
        yCen.value + 1.05 * R200c[i],
        r"$R_{200c}$",
        color="white",
        ha="center",
        va="bottom"
    )
    ax.text(
        xCen.value,
        yCen.value + 1.05 * 5*R200c[i],
        r"$5 \times R_{200c}$",
        color="grey",
        ha="center",
        va="bottom"
    )
    circle_r200 = plt.Circle((xCen, yCen), R200c[i], color="white", fill=False, linestyle='-')
    circle_5r200 = plt.Circle((xCen, yCen), 5*R200c[i], color="grey", fill=False, linestyle='--')
    ax.add_artist(circle_r200)
    ax.add_artist(circle_5r200)
    fig.savefig(f"outfiles/halo{author}{i}_DMmap{out_to_radius}r200_parent.png")
    plt.close(fig)
