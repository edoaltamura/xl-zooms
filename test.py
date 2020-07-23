# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import swiftsimio as sw
import unyt
from swiftsimio.visualisation.projection import project_pixel_grid
from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths

# EAGLE-XL data path
dataPath = "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/"
# VR data path
vrPath = dataPath + "stf_swiftdm_3dfof_subhalo_0036/"
# Halo properties file
haloPropFile = vrPath + "stf_swiftdm_3dfof_subhalo_0036.VELOCIraptor.properties.0"
# Snapshot file
snapFile = dataPath + "EAGLE-XL_L0300N0564_DMONLY_0036.hdf5"

# Define mask
x = [134.688, 90.671, 71.962]
y = [169.921, 289.822, 69.291]
z = [289.233, 98.227, 240.338]
choice = 0
rad = 5.2

xCen = unyt.unyt_quantity(x[choice], unyt.Mpc)
yCen = unyt.unyt_quantity(y[choice], unyt.Mpc)
zCen = unyt.unyt_quantity(z[choice], unyt.Mpc)
size = unyt.unyt_quantity(1. * rad, unyt.Mpc)

mask = sw.mask(snapFile)
box = mask.metadata.boxsize
region = [
    [xCen - size, xCen + size],
    [yCen - size, yCen + size],
    [zCen - size, zCen + size]
]
mask.constrain_spatial(region)
# Load data using mask
data = sw.load(snapFile, mask=mask)
posDM = data.dark_matter.coordinates

# print('Generating point particle map...')
# plt.figure()
# plt.axes().set_aspect('equal')
# plt.plot(posDM[:, 0] - xCen, posDM[:, 1] - yCen, ',')
# plt.xlim([-size, size])
# plt.ylim([-size, size])
# plt.savefig('particles.png')
# plt.show()
# plt.clf()


# Generate smoothing lengths for the dark matter
data.dark_matter.smoothing_lengths = generate_smoothing_lengths(
    data.dark_matter.coordinates,
    data.metadata.boxsize,
    kernel_gamma=1.8,
    neighbours=60,
    speedup_fac=2,
    dimension=3,
)

# Project the dark matter mass
dm_mass = project_pixel_grid(
    # Note here that we pass in the dark matter dataset not the whole
    # data object, to specify what particle type we wish to visualise
    data=data.dark_matter,
    boxsize=data.metadata.boxsize,
    resolution=1024,
    project=None,
    parallel=True,
    region=None
)

print('Generating smoothed DMO map...')
from matplotlib.pyplot import imsave
from matplotlib.colors import LogNorm

imsave("dm_mass_map.png", LogNorm()(dm_mass), cmap="inferno")
plt.show()
