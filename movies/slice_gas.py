import sys
import copy
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

sys.path.append("..")

from scaling_relations import SliceGas
from register import parser, default_output_directory

field = 'entropies'

parser.add_argument(
    '-s',
    '--snapshot-number',
    type=int,
    required=True
)
args = parser.parse_args()

dir = '/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_alpha1p0/'
s = dir + f"snapshots/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_{args.snapshot_number:04d}.hdf5"
c = dir + f"stf/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_{args.snapshot_number:04d}/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_{args.snapshot_number:04d}.properties"

gf = SliceGas(field)

try:
    slice = gf.process_single_halo(
        path_to_snap=s,
        path_to_catalogue=c,
        temperature_range=(1e5, 1e10),
        depth_offset=None,  # Goes through the centre of potential
        return_type='class'
    )
except Exception as e:
    print(f"Snap number {args.snapshot_number:04d} could not be processed.", e, sep='\n')

# Display
fig, axes = plt.subplots()

print(f"Min: {np.nanmin(slice.extremes[0]):.2E}\nMax: {np.nanmax(slice.extremes[1]):.2E}")

cmap = copy.copy(plt.get_cmap('twilight'))
cmap.set_under('black')
axes.axis("off")
axes.set_aspect("equal")
axes.imshow(
    slice.gas_map.T,
    norm=LogNorm(vmin=1E5, vmax=3E8),
    cmap=cmap,
    origin="lower",
    extent=slice.region
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
fig.savefig(
    os.path.join(
        default_output_directory,
        f"{field}_{os.path.basename(s)[:-5].replace('.', 'p')}.png"
    ),
    dpi=300
)

if not args.quiet:
    fig.set_tight_layout(False)
    plt.show()
