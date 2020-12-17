import unyt
import numpy as np
from multiprocessing import Pool, cpu_count
import h5py as h5
import swiftsimio as sw
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from register import (
    zooms_register,
    Zoom,
    Tcut_halogas,
    name_list,
    vr_numbers
)

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

# Constants
bins = 40
radius_bounds = [0.01, 6.]  # In units of R500crit
fbary = 0.15741  # Cosmic baryon fraction
mean_molecular_weight = 0.59
mean_atomic_weight_per_free_electron = 1.14


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def feedback_stats_dT(path_to_snap: str, path_to_catalogue: str) -> tuple:
    # Read in halo properties
    with h5.File(f'{path_to_catalogue}', 'r') as h5file:
        XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc)
        YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc)
        ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc)
        M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
        R500c = unyt.unyt_quantity(h5file['/SO_R_500_rhocrit'][0], unyt.Mpc)

    # Read in gas particles
    mask = sw.mask(f'{path_to_snap}', spatial_only=False)
    region = [[XPotMin - radius_bounds[1] * R500c, XPotMin + radius_bounds[1] * R500c],
              [YPotMin - radius_bounds[1] * R500c, YPotMin + radius_bounds[1] * R500c],
              [ZPotMin - radius_bounds[1] * R500c, ZPotMin + radius_bounds[1] * R500c]]
    mask.constrain_spatial(region)
    mask.constrain_mask("gas", "temperatures", Tcut_halogas * mask.units.temperature, 1.e12 * mask.units.temperature)
    data = sw.load(f'{path_to_snap}', mask=mask)

    feedback_stats, edges = np.histogram(np.log10(data.black_holes.feedback_delta_t.value), bins=30)

    return feedback_stats, edges[:-1]


def _process_single_halo(zoom: Zoom):
    return feedback_stats_dT(zoom.snapshot_file, zoom.catalog_file)


if __name__ == "__main__":
    vr_num = 'Minimum'

    zooms_register = [zoom for zoom in zooms_register if f"{vr_num}" in zoom.run_name]
    name_list = [zoom for zoom in name_list if f"{vr_num}" in zoom]

    # The results of the multiprocessing Pool are returned in the same order as inputs
    with Pool() as pool:
        print(f"Analysis mapped onto {cpu_count():d} CPUs.")
        results = pool.map(_process_single_halo, iter(zooms_register))

        # Recast output into a Pandas dataframe for further manipulation
        columns = [
            'num',
            'bin'
        ]
        results = pd.DataFrame(list(results), columns=columns)
        results.insert(0, 'run_name', pd.Series(name_list, dtype=str))
        print(results.head())

    fig, ax = plt.subplots()

    for i in range(len(results)):

        style = ''
        if '-8res' in results.loc[i, "run_name"]:
            style = ':'
        elif '+1res' in results.loc[i, "run_name"]:
            style = '-'

        color = ''
        if 'Ref' in results.loc[i, "run_name"]:
            color = 'black'
        elif 'MinimumDistance' in results.loc[i, "run_name"]:
            color = 'orange'
        elif 'Isotropic' in results.loc[i, "run_name"]:
            color = 'lime'

        ax.step(
            results['bin'][i],
            results['num'][i],
            linestyle=style, linewidth=1, color=color, alpha=1,
            # label=results.loc[i, "run_name"]
        )

    ax.set_yscale('log')
    ax.set_ylabel(r'Number of feedback events')
    ax.set_xlabel(r'$\log_{10}(\DeltaT)$')

    plt.legend()
    plt.show()
