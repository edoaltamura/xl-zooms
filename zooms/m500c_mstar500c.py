# Plot scaling relations for EAGLE-XL tests
import os
import unyt
import numpy as np
from typing import Tuple
from multiprocessing import Pool
import h5py as h5
import swiftsimio as sw
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from register import zooms_register, Zoom
import observational_data as obs

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


def _process_single_halo(zoom: Zoom):
    return process_single_halo(zoom.snapshot_file, zoom.catalog_file)


def make_single_image():
    fig, ax = plt.subplots()

    M500c = np.zeros(len(zooms_register), dtype=np.float64)
    Mstar500c = np.zeros(len(zooms_register), dtype=np.float64)
    Mhot500c = np.zeros(len(zooms_register), dtype=np.float64)

    print((
        f"{'Run name':<40s} "
        f"{'M_500crit':<15s} "
        f"{'M_star(< R_500crit)':<25s} "
        f"{'M_hot(< R_500crit)':<20s} "
    ))

    # The results of the multiprocessing Pool are returned in the same order as inputs
    with Pool() as pool:
        results = pool.map(_process_single_halo, iter(zooms_register))

    for i, data in enumerate(results):
        # `data` is a tuple with (M_500crit, M_hotgas, f_hotgas)
        # Results returned as tuples, which are immutable. Convert to list to update.
        data = list(data)

        h70_XL = H0_XL / 70.
        data[0] = data[0] * h70_XL
        data[1] = data[1] * (h70_XL ** 2.5)  # * 1.e10)
        M500c[i] = data[0].value
        Mstar500c[i] = data[1].value
        Mhot500c[i] = data[2].value

        print((
            f"{zooms_register[i].run_name:<40s} "
            f"{(data[0].value / 1.e13):<6.4f} * 1e13 Msun "
            f"{(data[1].value / 1.e13):<6.4f} * 1e13 Msun "
            f"{(data[2].value / 1.e13):<6.4f} "
        ))

    ax.scatter(M500c, Mstar500c, c=[zoom.plot_color for zoom in zooms_register], alpha=0.7, s=10, edgecolors='none')

    # Display observational data
    Budzynski14 = obs.Budzynski14()
    Kravtsov18 = obs.Kravtsov18()
    ax.plot(Budzynski14.M500, Budzynski14.Mstar500, linestyle='-', color='gray')
    ax.scatter(Kravtsov18.M500, Kravtsov18.Mstar500, marker='*', alpha=0.7, color='gray', edgecolors='none')

    ax.set_xlabel(r'$M_{500{\rm crit}}\ [{\rm M}_{\odot}]$')
    ax.set_ylabel(r'$M_{{\rm star},500{\rm crit}}\ [{\rm M}_{\odot}]$')

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Build legends
    handles = [
        Line2D([], [], color='black', marker='.', markeredgecolor='none', linestyle='None', markersize=6, label='1/8 EAGLE'),
        Line2D([], [], color='black', marker='^', markeredgecolor='none', linestyle='None', markersize=6, label='EAGLE'),
    ]
    plt.legend(handles=handles, title="Resolution")
    handles = [
        Patch(facecolor='black', edgecolor='None', label='Random (Ref)'),
        Patch(facecolor='orange', edgecolor='None', label='Minimum distance'),
        Patch(facecolor='lime', edgecolor='None', label='Isotropic'),
    ]
    plt.legend(handles=handles, title="AGN model")
    handles = [
        Line2D([], [], color='grey', marker='.', markeredgecolor='none', linestyle='-', markersize=0, label=Budzynski14.paper_name),
        Line2D([], [], color='grey', marker='*', markeredgecolor='none', linestyle='None', markersize=4, label=Kravtsov18.paper_name),
    ]
    plt.legend(handles=handles, title="Observations")
    fig.savefig(f'{zooms_register[0].output_directory}/m500c_mstar500c.png', dpi=300)
    plt.show()
    plt.close()

    return


make_single_image()

# def make_single_image(
#         name_list: List[str] = None,
#         paths_to_snap: List[str] = None,
#         paths_to_catalogue: List[str] = None,
#         output_path: str = None
# ) -> None:
#
#     assert len(paths_to_snap) == len(paths_to_catalogue)
#
#     numZooms = len(paths_to_snap)
#     M500c = np.zeros(numZooms)
#     Mstar500c = np.zeros(numZooms)
#     Mhot500c = np.zeros(numZooms)
#
#     for i, (snap, catalogue) in enumerate(zip(paths_to_snap, paths_to_catalogue)):
#         results = process_single_halo(snap, catalogue)
#         M500c[i] = results[0]
#         Mstar500c[i] = results[1]
#         Mhot500c[i] = results[2]
#
#     # Budzynski et al. 2014
#     M500_Bud = np.array([5., 100.])
#     Mstar500_Bud = 10. ** (0.89 * np.log10(M500_Bud / 30.) - 0.56)
#
#     # Kravtsov et al. 2018
#     M500_Kra = np.array([15.60, 10.30, 7.00, 5.34, 2.35, 1.86, 1.34, 0.46, 0.47]) * 10.
#     Mstar500_Kra = np.array([15.34, 12.35, 8.34, 5.48, 2.68, 3.48, 2.86, 1.88, 1.85]) * 0.1
#
#     h70_XL = H0_XL / 70.
#     M500c *= h70_XL
#     Mhot500c *= (h70_XL ** 2.5)
#     Mstar500c *= (h70_XL ** 2.5)
#
#     fig, ax = plt.subplots(figsize=(5, 3))
#     ax.plot(M500_Bud * 1.e13, Mstar500_Bud * 1.e13, linestyle='-', color='gray', label='Budzynski et al. (2014)')
#     ax.scatter(M500_Kra * 1.e13, Mstar500_Kra * 1.e13, marker='*', alpha=0.7, color='gray', label='Kravtsov et al. (2018)')
#
#     print(f"\n{'Run name':<25s} {'M500c             ':<15s} {'Mhot500c           ':<15s}")
#     for i in range(numZooms):
#         print(f"{name_list[i]:<25s} {(M500c[i] / 1.e13):<5.3f} * 1e13 Msun {(Mhot500c[i] / 1.e13):<5.3f} * 1e13 Msun")
#         ax.scatter(M500c[i], Mstar500c[i], c=colours[i], label=name_list[i], alpha=0.5, s=5)
#
#     ax.set_xlabel(r'$M_{500{\rm c}}/h_{70}^{-1}{\rm M}_{\odot}$')
#     ax.set_ylabel(r'$M_{{\rm star},500{\rm c}}/h_{70}^{-5/2}{\rm M}_{\odot}$')
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#     fig.savefig(f'{output_path}/m500c_mstar500c.png', dpi=500)
#     plt.show()
#     plt.close()
#
#     return