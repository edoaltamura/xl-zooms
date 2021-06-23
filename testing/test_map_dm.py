import sys
import copy
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

sys.path.append("..")

from scaling_relations import MapDM
from register import find_files, xlargs
# snap, cat = find_files()

dir = '/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_alpha1p0/'
snap = dir + f"snapshots/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_{xlargs.snapshot_number:04d}.hdf5"
cat = dir + f"stf/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_{xlargs.snapshot_number:04d}/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_{xlargs.snapshot_number:04d}.properties"

gf = MapDM(backend='renormalised')
map_dm = gf.process_single_halo(
    path_to_snap=snap,
    path_to_catalogue=cat,
    depth=1
)

# Display
cmap = copy.copy(plt.get_cmap('twilight'))
cmap.set_under('black')

fig, axes = plt.subplots()

axes.axis("off")
axes.set_aspect("equal")
cmap_bkgr = copy.copy(plt.get_cmap(cmap))
cmap_bkgr.set_under('black')
axes.imshow(
        map_dm.map.T,
        norm=LogNorm(),
        cmap=cmap_bkgr,
        origin="lower",
        extent=map_dm.region
    )
axes.text(
    0.025,
    0.025,
    'Masses',
    color="w",
    ha="left",
    va="bottom",
    alpha=0.8,
    transform=axes.transAxes,
)
plt.show()
