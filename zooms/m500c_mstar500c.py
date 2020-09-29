# Plot scaling relations for EAGLE-XL tests

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import swiftsimio as sw
import unyt
from typing import List, Tuple

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

THOT = 1.e5  # Hot gas temperature threshold in K

fbary = 0.15741  # Cosmic baryon fraction
H0_XL = 67.66  # Hubble constant in km/s/Mpc


def process_single_halo(
        path_to_snap: str,
        path_to_catalogue: str
) -> Tuple[float]:

    # Read in halo properties
    with h5.File(f'{path_to_catalogue}', 'r') as h5file:
        XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc)
        YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc)
        ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc)
        M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
        Mstar500c = unyt.unyt_quantity(h5file['/SO_Mass_star_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
        R500c = unyt.unyt_quantity(h5file['/SO_R_500_rhocrit'][0], unyt.Mpc)

    print(XPotMin, YPotMin, ZPotMin, M500c, R500c)

    # Read in gas particles
    mask = sw.mask(f'{path_to_snap}')
    region = [[XPotMin - R500c, XPotMin + R500c],
              [YPotMin - R500c, YPotMin + R500c],
              [ZPotMin - R500c, ZPotMin + R500c]]
    mask.constrain_spatial(region)
    data = sw.load(f'{path_to_snap}', mask=mask)
    posGas = data.gas.coordinates
    massGas = data.gas.masses
    tempGas = data.gas.temperatures

    # Select hot gas within sphere
    deltaX = posGas[:, 0] - XPotMin
    deltaY = posGas[:, 1] - YPotMin
    deltaZ = posGas[:, 2] - ZPotMin
    deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2)
    index = np.where((deltaR < R500c) & (tempGas > THOT))[0]
    Mhot500c = np.sum(massGas[index])

    return M500c, Mstar500c, Mhot500c


def make_single_image(
        name_list: List[str] = None,
        paths_to_snap: List[str] = None,
        paths_to_catalogue: List[str] = None,
        output_path: str = None
) -> None:

    assert len(paths_to_snap) == len(paths_to_catalogue)

    numZooms = len(paths_to_snap)
    M500c = np.zeros(numZooms)
    Mstar500c = np.zeros(numZooms)
    Mhot500c = np.zeros(numZooms)

    for i, (snap, catalogue) in enumerate(zip(paths_to_snap, paths_to_catalogue)):
        results = process_single_halo(snap, catalogue)
        M500c[i] = results[0]
        Mstar500c[i] = results[1]
        Mhot500c[i] = results[2]

    # Budzynski et al. 2014
    M500_Bud = np.array([5., 100.])
    Mstar500_Bud = 10. ** (0.89 * np.log10(M500_Bud / 30.) - 0.56)

    # Kravtsov et al. 2018
    M500_Kra = np.array([15.60, 10.30, 7.00, 5.34, 2.35, 1.86, 1.34, 0.46, 0.47]) * 10.
    Mstar500_Kra = np.array([15.34, 12.35, 8.34, 5.48, 2.68, 3.48, 2.86, 1.88, 1.85]) * 0.1

    h70_XL = H0_XL / 70.
    M500c *= h70_XL
    Mhot500c *= (h70_XL ** 2.5)
    Mstar500c *= (h70_XL ** 2.5)

    colours = [
        'blue', 'blue', 'blue', 'cyan',
        'purple', 'purple', 'purple', 'red',
        'orange', 'orange', 'orange', 'yellow',
        'green', 'green', 'green', 'lime',
        'brown'
    ]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(M500_Bud * 1.e13, Mstar500_Bud * 1.e13, linestyle='-', color='gray', label='Budzynski et al. (2014)')
    ax.scatter(M500_Kra * 1.e13, Mstar500_Kra * 1.e13, marker='*', alpha=0.7, color='gray', label='Kravtsov et al. (2018)')

    print(f"\n{'Run name':<25s} {'M500c             ':<15s} {'Mhot500c           ':<15s}")
    for i in range(numZooms):
        print(f"{name_list[i]:<25s} {(M500c[i] / 1.e13):<5.3f} * 1e13 Msun {(Mhot500c[i] / 1.e13):<5.3f} * 1e13 Msun")
        ax.scatter(M500c[i], Mstar500c[i], c=colours[i], label=name_list[i], alpha=0.5, s=5)

    ax.set_xlabel(r'$M_{500{\rm c}}/h_{70}^{-1}{\rm M}_{\odot}$')
    ax.set_ylabel(r'$M_{{\rm star},500{\rm c}}/h_{70}^{-5/2}{\rm M}_{\odot}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    fig.savefig(f'{output_path}/m500c_mstar500c.png', dpi=500)
    plt.show()
    plt.close()

    return


if __name__ == "__main__":

    name_list = [
        "SK0_-8res_AGN1",
        "SK1_-8res_AGN1",
        "SK2_-8res_AGN1",
        "SK0_+1res_AGN1",

        "SK0_-8res_AGN1_AGNseed1e4",
        "SK1_-8res_AGN1_AGNseed1e4",
        "SK2_-8res_AGN1_AGNseed1e4",
        "SK0_+1res_AGN8_AGNseed1e4",

        "SK0_-8res_AGN1",
        "SK1_-8res_AGN1",
        "SK2_-8res_AGN1",
        "SK0_+1res_AGN1",

        "SK0_-8res_AGN1_DefSep",
        "SK1_-8res_AGN1_DefSep",
        "SK2_-8res_AGN1_DefSep",
        "SK0_+1res_AGN1_DefSep",
        "SK0_+1res_AGN8_DefSep",
    ]
    snapshot_filenames = [
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK0_-8res/snapshots/EAGLE-XL_ClusterSK0_-8res_0036.hdf5",
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK1_-8res/snapshots/EAGLE-XL_ClusterSK1_-8res_0036.hdf5",
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK2_-8res/snapshots/EAGLE-XL_ClusterSK2_-8res_0036.hdf5",
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK0_+1res/snapshots/EAGLE-XL_ClusterSK0_+1res_0036.hdf5",

        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK0_-8res_AGNseedmass1e4/snapshots/snap_2749.hdf5",
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK1_-8res_AGNseedmass1e4/snapshots/snap_2749.hdf5",
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK2_-8res_AGNseedmass1e4/snapshots/snap_2749.hdf5",
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK0_+1res_AGNseedmass1e4/snapshots/snap_2749.hdf5",

        "/cosma7/data/dp004/stk/SwiftRuns/EAGLE-XL/GroupZooms/ClusterSK0_-8res/snapshots/EAGLE-XL_ClusterSK0_HYDRO_0036.hdf5",
        "/cosma7/data/dp004/stk/SwiftRuns/EAGLE-XL/GroupZooms/ClusterSK1_-8res/snapshots/EAGLE-XL_ClusterSK1_HYDRO_0036.hdf5",
        "/cosma7/data/dp004/stk/SwiftRuns/EAGLE-XL/GroupZooms/ClusterSK2_-8res/snapshots/EAGLE-XL_ClusterSK2_HYDRO_0036.hdf5",
        "/cosma7/data/dp004/stk/SwiftRuns/EAGLE-XL/GroupZooms/ClusterSK0_+1res/snapshots/EAGLE-XL_ClusterSK0_HYDRO_0036.hdf5",

        "/cosma7/data/dp004/stk/SwiftRuns/EAGLE-XL/GroupZooms/ClusterSK0_-8res_DefSep/snapshots/EAGLE-XL_ClusterSK0_HYDRO_0036.hdf5",
        "/cosma7/data/dp004/stk/SwiftRuns/EAGLE-XL/GroupZooms/ClusterSK1_-8res_DefSep/snapshots/EAGLE-XL_ClusterSK1_HYDRO_0036.hdf5",
        "/cosma7/data/dp004/stk/SwiftRuns/EAGLE-XL/GroupZooms/ClusterSK2_-8res_DefSep/snapshots/EAGLE-XL_ClusterSK2_HYDRO_0036.hdf5",
        "/cosma7/data/dp004/stk/SwiftRuns/EAGLE-XL/GroupZooms/ClusterSK0_+1res_DefSep/snapshots/EAGLE-XL_ClusterSK0_HYDRO_0036.hdf5",
        "/cosma7/data/dp004/stk/SwiftRuns/EAGLE-XL/GroupZooms/ClusterSK0_+1res_DefSep_AGN8/snapshots/EAGLE-XL_ClusterSK0_HYDRO_0036.hdf5",
    ]
    catalogue_filenames = [
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK0_-8res/stf/EAGLE-XL_ClusterSK0_-8res_0036/EAGLE-XL_ClusterSK0_-8res_0036.properties",
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK1_-8res/stf/EAGLE-XL_ClusterSK1_-8res_0036/EAGLE-XL_ClusterSK1_-8res_0036.properties",
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK2_-8res/stf/EAGLE-XL_ClusterSK2_-8res_0036/EAGLE-XL_ClusterSK2_-8res_0036.properties",
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK0_+1res/stf/EAGLE-XL_ClusterSK0_+1res_0036/EAGLE-XL_ClusterSK0_+1res_0036.properties",

        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK0_-8res_AGNseedmass1e4/stf/snap_2749/snap_2749.properties",
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK1_-8res_AGNseedmass1e4/stf/snap_2749/snap_2749.properties",
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK2_-8res_AGNseedmass1e4/stf/snap_2749/snap_2749.properties",
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK0_+1res_AGNseedmass1e4/stf/snap_2749/snap_2749.properties",

        "/cosma7/data/dp004/stk/SwiftRuns/EAGLE-XL/GroupZooms/ClusterSK0_-8res/stf/EAGLE-XL_ClusterSK0_HYDRO_0036/EAGLE-XL_ClusterSK0_HYDRO_0036.properties",
        "/cosma7/data/dp004/stk/SwiftRuns/EAGLE-XL/GroupZooms/ClusterSK1_-8res/stf/EAGLE-XL_ClusterSK1_HYDRO_0036/EAGLE-XL_ClusterSK1_HYDRO_0036.properties",
        "/cosma7/data/dp004/stk/SwiftRuns/EAGLE-XL/GroupZooms/ClusterSK2_-8res/stf/EAGLE-XL_ClusterSK2_HYDRO_0036/EAGLE-XL_ClusterSK2_HYDRO_0036.properties",
        "/cosma7/data/dp004/stk/SwiftRuns/EAGLE-XL/GroupZooms/ClusterSK0_+1res/stf/EAGLE-XL_ClusterSK0_HYDRO_0036/EAGLE-XL_ClusterSK0_HYDRO_0036.properties",

        "/cosma7/data/dp004/stk/SwiftRuns/EAGLE-XL/GroupZooms/ClusterSK0_-8res_DefSep/stf/EAGLE-XL_ClusterSK0_HYDRO_0036/EAGLE-XL_ClusterSK0_HYDRO_0036.properties",
        "/cosma7/data/dp004/stk/SwiftRuns/EAGLE-XL/GroupZooms/ClusterSK1_-8res_DefSep/stf/EAGLE-XL_ClusterSK1_HYDRO_0036/EAGLE-XL_ClusterSK1_HYDRO_0036.properties",
        "/cosma7/data/dp004/stk/SwiftRuns/EAGLE-XL/GroupZooms/ClusterSK2_-8res_DefSep/stf/EAGLE-XL_ClusterSK2_HYDRO_0036/EAGLE-XL_ClusterSK2_HYDRO_0036.properties",
        "/cosma7/data/dp004/stk/SwiftRuns/EAGLE-XL/GroupZooms/ClusterSK0_+1res_DefSep/stf/EAGLE-XL_ClusterSK0_HYDRO_0036/EAGLE-XL_ClusterSK0_HYDRO_0036.properties",
        "/cosma7/data/dp004/stk/SwiftRuns/EAGLE-XL/GroupZooms/ClusterSK0_+1res_DefSep_AGN8/stf/EAGLE-XL_ClusterSK0_HYDRO_0036/EAGLE-XL_ClusterSK0_HYDRO_0036.properties",
    ]
    output_directory = "/cosma7/data/dp004/dc-alta2/xl-zooms/analysis"


    make_single_image(
        name_list=name_list,
        paths_to_snap=snapshot_filenames,
        paths_to_catalogue=catalogue_filenames,
        output_path=output_directory,
    )
