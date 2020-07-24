import numpy as np
import unyt
import swiftsimio as sw
from swiftsimio.visualisation.projection import project_pixel_grid
from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from utils import (
    latex_float
)

print("Loading halos selected...")
lines = np.loadtxt("halo_selected.txt", comments="#", delimiter=",", unpack=False).T
print("log10(M200c / Msun): ", np.log10(lines[1]*1e13))
print("R200c: ", lines[2])
print("Centre of potential coordinates: (xC, yC, zC)")
for i in range(3):
    print(f"\tHalo {i:d}:\t({lines[3,i]:2.1f}, {lines[4,i]:2.1f}, {lines[5,i]:2.1f})")
M200c = lines[1]*1e13
R200c = lines[2]
x = lines[3]
y = lines[4]
z = lines[5]

def dm_render(swio_data, region: list = None):
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
        resolution=1024,
        project=None,
        parallel=True,
        region=region
    )
    return dm_map

print("\nRendering snapshot volume...")
# EAGLE-XL data path
dataPath = "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/"
snapFile = dataPath + "EAGLE-XL_L0300N0564_DMONLY_0036.hdf5"
data = sw.load(snapFile)

dm_mass = dm_render(data)
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
fig.savefig(f"volume_DMmap.png")
plt.close(fig)


for i in range(3):
    xCen = unyt.unyt_quantity(x[i], unyt.Mpc)
    yCen = unyt.unyt_quantity(y[i], unyt.Mpc)
    zCen = unyt.unyt_quantity(z[i], unyt.Mpc)
    size = unyt.unyt_quantity(R200c[i], unyt.Mpc)

    mask = sw.mask(snapFile)
    region = [
        [xCen - size, xCen + size],
        [yCen - size, yCen + size],
        [zCen - size, zCen + size]
    ]
    mask.constrain_spatial(region)
    # Load data using mask
    data = sw.load(snapFile, mask=mask)

    print(f"Rendering halo {i}...")
    dm_mass = dm_render(data, region=region)

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
    ax.text(
        0.975,
        0.025,
        (
            f"$M_{{200c}}={latex_float(M200c)}$ ${(unyt.Msun).units.latex_repr}$"
        ),
        color="white",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )
    fig.savefig(f"halo{i}_DMmap.png")
    plt.close(fig)

