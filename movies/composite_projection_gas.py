"""
Run with:
    nohup parallel python3 composite_slice_gas.py -q -s ::: {1..2522} &
"""

import sys
import copy
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

sys.path.append("..")

from scaling_relations import MapGas
from register import default_output_directory, find_files, xlargs
from literature import Cosmology

snap, cat = find_files()


def draw_panel(axes, field, cmap: str = 'Greys_r', vmin=None, vmax=None):
    gf = MapGas(field)

    try:
        slice = gf.process_single_halo(
            path_to_snap=snap,
            path_to_catalogue=cat,
            temperature_range=(1e5, 1e9),
        )

    except Exception as e:
        print(
            f"Snap number {xlargs.snapshot_number:04d} could not be processed.",
            e,
            sep='\n'
        )

    if xlargs.debug:
        print(f"{field.title()} map: min = {np.nanmin(slice.map):.2E}, max = {np.nanmax(slice.map):.2E}")
        print(f"Map centre: {[float(f'{i.v:.3f}') for i in slice.centre]} Mpc")

    cmap_bkgr = copy.copy(plt.get_cmap(cmap))
    cmap_bkgr.set_under('black')
    axes.axis("off")
    axes.set_aspect("equal")
    axes.imshow(
        slice.map.T,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap=cmap_bkgr,
        origin="lower",
        extent=slice.region
    )
    # axes.text(
    #     0.025,
    #     0.025,
    #     (
    #         f'{field.title()}\n'
    #         f'z = {slice.z:.2f}\n'
    #         f't = {Cosmology().age(slice.z).value:.3f} Gyr'
    #     ),
    #     color="w",
    #     ha="left",
    #     va="bottom",
    #     alpha=0.8,
    #     transform=axes.transAxes,
    # )


fig = plt.figure(figsize=(9, 3), constrained_layout=True)
gs = fig.add_gridspec(1, 3, hspace=0.01, wspace=0.01)
axes = gs.subplots()

draw_panel(axes[0], 'densities', cmap='bone', vmin=1E5, vmax=1E15)
draw_panel(axes[1], 'temperatures', cmap='twilight', vmin=1E5, vmax=2E8)
draw_panel(axes[2], 'entropies', cmap='inferno', vmin=1E7, vmax=1E9)

fig.savefig(
    os.path.join(
        default_output_directory,
        f"projection_composite_{os.path.basename(snap)[:-5].replace('.', 'p')}.png"
    ),
    dpi=300
)

if not xlargs.quiet:
    plt.show()
