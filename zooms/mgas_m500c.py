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
    fhot500c = Mhot500c / M500c

    return M500c, Mhot500c, fhot500c


def make_single_image(
        name_list: List[str] = None,
        paths_to_snap: List[str] = None,
        paths_to_catalogue: List[str] = None,
        output_path: str = None
) -> None:

    assert len(paths_to_snap) == len(paths_to_catalogue)

    numZooms = len(paths_to_snap)
    M500c = np.zeros(numZooms)
    Mhot500c = np.zeros(numZooms)
    fhot500c = np.zeros(numZooms)

    for i, (snap, catalogue) in enumerate(zip(paths_to_snap, paths_to_catalogue)):
        results = process_single_halo(snap, catalogue)
        M500c[i] = results[0]
        Mhot500c[i] = results[1]
        fhot500c[i] = results[2]

    # Sun et al. 2009
    M500_Sun = np.array(
        [3.18, 4.85, 3.90, 1.48, 4.85, 5.28, 8.49, 10.3, 2.0, 7.9, 5.6, 12.9, 8.0, 14.1, 3.22, 14.9, 13.4, 6.9, 8.95,
         8.8, 8.3, 9.7, 7.9]
    )
    f500_Sun = np.array(
        [0.097, 0.086, 0.068, 0.049, 0.069, 0.060, 0.076, 0.081, 0.108, 0.086, 0.056, 0.076, 0.075, 0.114, 0.074, 0.088,
         0.094, 0.094, 0.078, 0.099, 0.065, 0.090, 0.093]
    )
    Mgas500_Sun = M500_Sun * f500_Sun

    # Lovisari et al. 2015 (in h_70 units already)
    M500_Lov = np.array(
        [2.07, 4.67, 2.39, 2.22, 2.95, 2.83, 3.31, 3.53, 3.49, 3.35, 14.4, 2.34, 4.78, 8.59, 9.51, 6.96, 10.8, 4.37,
         8.00, 12.1]
    )
    Mgas500_Lov = np.array(
        [0.169, 0.353, 0.201, 0.171, 0.135, 0.272, 0.171, 0.271, 0.306, 0.247, 1.15, 0.169, 0.379, 0.634, 0.906, 0.534,
         0.650, 0.194, 0.627, 0.817]
    )

    # Convert units
    h70_Sun = 73. / 70.
    M500_Sun *= h70_Sun
    Mgas500_Sun *= (h70_Sun ** 2.5)

    h70_XL = H0_XL / 70.
    M500c *= h70_XL
    Mhot500c *= ((h70_XL ** 2.5) / 1.e3)

    colours = ['blue', 'blue', 'blue', 'cyan', 'purple', 'purple', 'purple', 'red']
    # shapes = ['s', 's', 's', 's', 'o', 'o']

    plt.figure()
    plt.scatter(M500_Sun * 1.e13, Mgas500_Sun, marker='s', s=5, alpha=0.7, c='gray', label='Sun et al. (2009)')
    plt.scatter(M500_Lov * 1.e13, Mgas500_Lov, marker='*', s=5, alpha=0.7, c='gray', label='Lovisari et al. (2015)')

    for i in range(numZooms):
        plt.scatter(M500c[i], Mhot500c[i], c=colours[i], label=name_list[i], alpha=0.5, s=3)

    plt.xlabel(r'$M_{500{\rm c}}/h_{70}^{-1}{\rm M}_{\odot}$')
    plt.ylabel(r'$M_{{\rm gas},500{\rm c}}/10^{13}h_{70}^{-5/2}{\rm M}_{\odot}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='lower right')
    # plt.plot(massRange,fbary*massRange,'--')
    plt.savefig(f'{output_path}/m500cgas_mhotgas.png', dpi=400)
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
    ]
    catalogue_filenames = [
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK0_-8res/stf/EAGLE-XL_ClusterSK0_-8res_0036/EAGLE-XL_ClusterSK0_-8res_0036.properties",
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK1_-8res/stf/EAGLE-XL_ClusterSK1_-8res_0036/EAGLE-XL_ClusterSK1_-8res_0036.properties",
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK2_-8res/stf/EAGLE-XL_ClusterSK2_-8res_0036/EAGLE-XL_ClusterSK2_-8res_0036.properties",
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK0_+1res/stf/EAGLE-XL_ClusterSK0_+1res_0036/EAGLE-XL_ClusterSK0_+1res_0036.properties",

        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK0_-8res_AGNseedmass1e4/stf/snap_2749/snap_2749.properties",
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK1_-8res_AGNseedmass1e4/stf/snap_2749/snap_2749.properties",
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK2_-8res_AGNseedmass1e4/stf/snap_2749/snap_2749.properties",
        "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/SK0_+1res_AGNseedmass1e4/stf/snap_2749/snap_2749.properties"
    ]
    output_directory = "/cosma7/data/dp004/dc-alta2/xl-zooms/analysis"


    make_single_image(
        name_list=name_list,
        paths_to_snap=snapshot_filenames,
        paths_to_catalogue=catalogue_filenames,
        output_path=output_directory,
    )
