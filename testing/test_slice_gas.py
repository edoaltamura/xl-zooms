import sys
import copy
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

sys.path.append("..")

from scaling_relations import SliceGas
from test_files import cat, snap

field = 'entropies'

gf = SliceGas(field)
slice = gf.process_single_halo(
    path_to_snap=snap,
    path_to_catalogue=cat,
    temperature_range=(1e5, 1e9),
    depth_offset=None,
    return_type='class'
)
slice_gas, region = slice.map, slice.region
print(slice_gas, region)

# Display
cmap = copy.copy(plt.get_cmap('twilight'))
cmap.set_under('black')

fig, axes = plt.subplots()

axes.axis("off")
axes.set_aspect("equal")
axes.imshow(
    slice_gas.T,
    norm=LogNorm(),
    cmap=cmap,
    origin="lower",
    extent=region
)
axes.text(
    0.025,
    0.025,
    field.title(),
    color="w",
    ha="left",
    va="bottom",
    alpha=0.8,
    transform=axes.transAxes,
)
plt.show()
