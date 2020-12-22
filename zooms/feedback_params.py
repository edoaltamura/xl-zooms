import unyt
import numpy as np
from collections import defaultdict
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
    vr_numbers,
    get_allpaths_from_last
)

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

# Constants
bins = 40
radius_bounds = [0.01, 1.]  # In units of R500crit
fbary = 0.15741  # Cosmic baryon fraction
mean_molecular_weight = 0.59
mean_atomic_weight_per_free_electron = 1.14

BH_LOCK = 'id'  # 'cop'


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def feedback_stats_dT(path_to_snap: str, path_to_catalogue: str) -> dict:
    # Read in halo properties
    with h5.File(f'{path_to_catalogue}', 'r') as h5file:
        XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc)
        YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc)
        ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc)
        M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
        R500c = unyt.unyt_quantity(h5file['/SO_R_500_rhocrit'][0], unyt.Mpc)

    # Read in particles
    mask = sw.mask(f'{path_to_snap}', spatial_only=True)
    region = [[XPotMin - radius_bounds[1] * R500c, XPotMin + radius_bounds[1] * R500c],
              [YPotMin - radius_bounds[1] * R500c, YPotMin + radius_bounds[1] * R500c],
              [ZPotMin - radius_bounds[1] * R500c, ZPotMin + radius_bounds[1] * R500c]]
    mask.constrain_spatial(region)
    data = sw.load(f'{path_to_snap}', mask=mask)

    # Get positions for all BHs in the bounding region
    bh_positions = data.black_holes.coordinates
    bh_coordX = bh_positions[:, 0] - XPotMin
    bh_coordY = bh_positions[:, 1] - YPotMin
    bh_coordZ = bh_positions[:, 2] - ZPotMin
    bh_radial_distance = np.sqrt(bh_coordX ** 2 + bh_coordY ** 2 + bh_coordZ ** 2)

    # Get the central BH closest to centre of halo at z=0
    central_bh = defaultdict(list)
    central_bh_index = np.argmin(bh_radial_distance)
    central_bh['x'] = [bh_coordX[central_bh_index]]
    central_bh['y'] = [bh_coordY[central_bh_index]]
    central_bh['z'] = [bh_coordZ[central_bh_index]]
    central_bh['r'] = [bh_radial_distance[central_bh_index]]
    central_bh['mass'] = [data.black_holes.dynamical_masses[central_bh_index]]
    central_bh['id'] = [data.black_holes.particle_ids[central_bh_index]]
    central_bh['redshift'] = [data.metadata.z]

    # Retrieve BH data from other snaps
    all_snaps = get_allpaths_from_last(path_to_snap)
    all_catalogues = get_allpaths_from_last(path_to_catalogue)
    assert len(all_snaps) == len(all_catalogues), (
        f"Detected different number of high-z snaps and high-z catalogues. "
        f"Number of snaps: {len(all_snaps)}. Number of catalogues: {len(all_catalogues)}."
    )
    for highz_snap, highz_catalogue in zip(all_snaps[::-1], all_catalogues[::-1]):

        if (
                highz_snap != path_to_snap and
                highz_catalogue != path_to_catalogue and
                sw.load(f'{highz_snap}').metadata.z < 3.
        ):
            # Do not repeat redshift zero and keep only z < 3
            print(f"Analysing:\n\t{highz_snap}\n\t{highz_catalogue}")

            with h5.File(f'{highz_catalogue}', 'r') as h5file:
                XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc)
                YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc)
                ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc)
                M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
                R500c = unyt.unyt_quantity(h5file['/SO_R_500_rhocrit'][0], unyt.Mpc)

            data = sw.load(highz_snap)
            bh_positions = data.black_holes.coordinates
            bh_coordX = (bh_positions[:, 0] - XPotMin) / data.metadata.a
            bh_coordY = (bh_positions[:, 1] - YPotMin) / data.metadata.a
            bh_coordZ = (bh_positions[:, 2] - ZPotMin) / data.metadata.a
            bh_radial_distance = np.sqrt(bh_coordX ** 2 + bh_coordY ** 2 + bh_coordZ ** 2)

            if BH_LOCK == 'id':
                central_bh_index = np.where(data.black_holes.particle_ids.v == central_bh['id'][0].v)[0]
            elif BH_LOCK == 'cop':
                central_bh_index = np.argmin(bh_radial_distance)

            central_bh['x'].append(bh_coordX[central_bh_index])
            central_bh['y'].append(bh_coordY[central_bh_index])
            central_bh['z'].append(bh_coordZ[central_bh_index])
            central_bh['r'].append(bh_radial_distance[central_bh_index])
            central_bh['mass'].append(data.black_holes.dynamical_masses[central_bh_index])
            central_bh['id'].append(data.black_holes.particle_ids[central_bh_index])
            central_bh['redshift'].append(data.metadata.z)

    for key in central_bh:
        central_bh[key] = sw.cosmo_array(central_bh[key])

    return central_bh


def _process_single_halo(zoom: Zoom):
    return feedback_stats_dT(zoom.snapshot_file, zoom.catalog_file)


if __name__ == "__main__":
    vr_num = 'Minimum'

    zooms_register = [zoom for zoom in zooms_register if f"{vr_num}" in zoom.run_name]
    name_list = [zoom for zoom in name_list if f"{vr_num}" in zoom]

    central_bh = _process_single_halo(zooms_register[0])
    print(central_bh)

    # The results of the multiprocessing Pool are returned in the same order as inputs
    # with Pool() as pool:
    #     print(f"Analysis mapped onto {cpu_count():d} CPUs.")
    #     results = pool.map(_process_single_halo, iter(zooms_register))
    #
    #     # Recast output into a Pandas dataframe for further manipulation
    #     columns = [
    #         'num',
    #         'bin'
    #     ]
    #     results = pd.DataFrame(list(results), columns=columns)
    #     results.insert(0, 'run_name', pd.Series(name_list, dtype=str))
    #     print(results.head())
    #
    # fig, ax = plt.subplots()
    #
    # for i in range(len(results)):
    #
    #     style = ''
    #     if '-8res' in results.loc[i, "run_name"]:
    #         style = ':'
    #     elif '+1res' in results.loc[i, "run_name"]:
    #         style = '-'
    #
    #     color = ''
    #     if 'Ref' in results.loc[i, "run_name"]:
    #         color = 'black'
    #     elif 'MinimumDistance' in results.loc[i, "run_name"]:
    #         color = 'orange'
    #     elif 'Isotropic' in results.loc[i, "run_name"]:
    #         color = 'lime'
    #
    #     ax.step(
    #         results['bin'][i],
    #         results['num'][i],
    #         linestyle=style, linewidth=1, color=color, alpha=1,
    #         # label=results.loc[i, "run_name"]
    #     )
    #
    # ax.set_yscale('log')
    # ax.set_ylabel(r'Number of feedback events')
    # ax.set_xlabel(r'$\log_{10}(\Delta T)$')
    #
    # plt.legend()
    # plt.show()
