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

from scaling_relations import SliceGas
from register import default_output_directory, find_files, xlargs, set_mnras_stylesheet
from literature import Cosmology

s, c = find_files()
set_mnras_stylesheet()


centres = np.load('map_centre_L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_alpha1p0.npy')

def draw_radius_contours(axes, resolution, region, r500, levels=[1.], color='green', r500_units=True, use_labels=True):
    # Make the norm object to define the image stretch
    x_bins, y_bins = np.meshgrid(
        np.linspace(region[0], region[1], resolution),
        np.linspace(region[2], region[3], resolution)
    )
    cylinder_function = np.sqrt(x_bins.flatten() ** 2 + y_bins.flatten() ** 2)
    cylinder_function = cylinder_function.reshape((resolution, resolution))
    _levels = [radius * r500 for radius in levels] if r500_units else levels
    _units = r'$r_{500}$' if r500_units else 'Mpc'

    contours = axes.contour(
        x_bins,
        y_bins,
        cylinder_function,
        _levels,
        colors=color,
        linewidths=0.3,
        alpha=0.5
    )

    if use_labels:
        fmt = {value: f'${value:.1f}$ {_units}' for value in levels}

        # work with logarithms for loglog scale
        # middle of the figure:
        xmin, xmax, ymin, ymax = axes.axis()
        mid = (xmin + xmax) / 2, (ymin + ymax) / 2

        label_pos = []
        i = 0
        for line in contours.collections:
            for path in line.get_paths():
                logvert = path.vertices
                i += 1

                # find closest point
                logdist = np.linalg.norm(logvert - mid, ord=2, axis=1)
                min_ind = np.argmin(logdist)
                label_pos.append(logvert[min_ind, :])

        # Draw contour labels
        axes.clabel(
            contours,
            inline=True,
            inline_spacing=3,
            rightside_up=True,
            colors=color,
            fontsize=5,
            fmt=fmt,
            manual=label_pos,
        )


def draw_panel(axes, field, cmap: str = 'Greys_r', vmin=None, vmax=None):
    # gf = SliceGas(field, resolution=1024)
    #
    # try:
    #     slice = gf.process_single_halo(
    #         path_to_snap=s,
    #         path_to_catalogue=c,
    #         temperature_range=(1e5, 1e9),
    #         depth_offset=None,  # Goes through the centre of potential
    #         mask_radius_r500=4
    #         # map_centre=centres[xlargs.snapshot_number, :-1]
    #     )
    #
    # except Exception as e:
    #     print(
    #         f"Snap number {xlargs.snapshot_number:04d} could not be processed.",
    #         e,
    #         sep='\n'
    #     )
    #
    # if xlargs.debug:
    #     print(f"{field.title()} map: min = {np.nanmin(slice.map):.2E}, max = {np.nanmax(slice.map):.2E}")
    #     print(f"Map centre: {[float(f'{i.v:.3f}') for i in slice.centre]} Mpc")
    #
    # cmap_bkgr = copy.copy(plt.get_cmap(cmap))
    # cmap_bkgr.set_under('black')
    # axes.axis("off")
    # axes.set_aspect("equal")
    # axes.imshow(
    #     slice.map.T,
    #     norm=LogNorm(vmin=vmin, vmax=vmax),
    #     cmap=cmap_bkgr,
    #     origin="lower",
    #     extent=slice.region
    # )
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
    draw_radius_contours(axes, resolution=1024, region=[-10, 10, -10, 10], r500=5, levels=[1.])


fig = plt.figure(figsize=(9, 3), constrained_layout=True)
gs = fig.add_gridspec(1, 3, hspace=0.01, wspace=0.01)
axes = gs.subplots()

draw_panel(axes[0], 'densities', cmap='bone', vmin=1E6, vmax=1E14)
draw_panel(axes[1], 'temperatures', cmap='twilight', vmin=1E4, vmax=2E8)
draw_panel(axes[2], 'entropies', cmap='inferno', vmin=1E5, vmax=1E9)

fig.savefig(
    os.path.join(
        default_output_directory,
        f"slice_composite_{os.path.basename(s)[:-5].replace('.', 'p')}.png"
    ),
    dpi=300
)

if not xlargs.quiet:
    plt.show()
