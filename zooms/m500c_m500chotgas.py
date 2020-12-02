# Plot scaling relations for EAGLE-XL tests

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import h5py as h5
import swiftsimio as sw
import unyt
from typing import List, Tuple

from register import zooms_register

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

Tcut_halogas = 1.e5  # Hot gas temperature threshold in K

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

    # print(XPotMin, YPotMin, ZPotMin, M500c, R500c)

    # Read in gas particles
    mask = sw.mask(f'{path_to_snap}', spatial_only=False)
    region = [[XPotMin - R500c, XPotMin + R500c],
              [YPotMin - R500c, YPotMin + R500c],
              [ZPotMin - R500c, ZPotMin + R500c]]
    mask.constrain_spatial(region)
    mask.constrain_mask("gas", "temperatures", Tcut_halogas * mask.units.temperature)
    data = sw.load(f'{path_to_snap}', mask=mask)
    posGas = data.gas.coordinates
    massGas = data.gas.masses

    # Select hot gas within sphere
    deltaX = posGas[:, 0] - XPotMin
    deltaY = posGas[:, 1] - YPotMin
    deltaZ = posGas[:, 2] - ZPotMin
    deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2)
    index = np.where(deltaR < R500c)[0]
    Mhot500c = np.sum(massGas[index])
    fhot500c = Mhot500c / M500c

    return M500c, Mhot500c, fhot500c


def make_single_image():
    fig, ax = plt.subplots()

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

    print((
        f"{'Run name':<25s} "
        f"{'M_500crit':<20s} "
        f"{'M_hotgas(< R_500crit)':<20s} "
        f"{'f_hotgas(< R_500crit)':<20s} "
    ))
    for i, zoom in enumerate(zooms_register):
        # `results` is a tuple with (M_500crit, M_hotgas, f_hotgas)
        results = process_single_halo(zoom.snapshot_file, zoom.catalog_file)
        results = list(results)

        h70_XL = H0_XL / 70.
        results[0] *= h70_XL
        results[0] *= ((h70_XL ** 2.5) * 1.e10)

        ax.scatter(results[0], results[1], c=zoom.plot_color, label=zoom.run_name[i], alpha=0.7, s=10,
                   edgecolors='none')
        print((
            f"{zoom.run_name:<25s} "
            f"{(results[0] / 1.e13):<5.2f} * 1e13 Msun "
            f"{(results[1] / 1.e13):<5.2f} * 1e13 Msun "
            f"{(results[2] / 1.e13):<5.2f} "
        ))

    ax.scatter(M500_Sun * 1.e13, Mgas500_Sun * 1.e13, marker='s', s=5, alpha=0.7, c='gray', label='Sun et al. (2009)',
               edgecolors='none')
    ax.scatter(M500_Lov * 1.e13, Mgas500_Lov * 1.e13, marker='*', s=5, alpha=0.7, c='gray',
               label='Lovisari et al. (2015)', edgecolors='none')

    ax.set_xlabel(r'$M_{500{\rm c}}/h_{70}^{-1}{\rm M}_{\odot}$')
    ax.set_ylabel(r'$M_{{\rm gas},500{\rm c}}/h_{70}^{-5/2}{\rm M}_{\odot}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(ax.get_xlim(), [lim * fbary for lim in ax.get_xlim()], '--', color='k')

    # Build legend
    handles = [
        mlines.Line2D([], [], color='black', marker='.', linestyle='None', markersize=10, label='Random AGN (Ref)'),
        mlines.Line2D([], [], color='orange', marker='.', linestyle='None', markersize=10, label='MinimumDistance'),
        mlines.Line2D([], [], color='lime', marker='.', linestyle='None', markersize=10, label='Isotropic')
    ]
    plt.legend(handles=handles)
    fig.savefig(f'{zooms_register[0].output_directory}/m500c_mhotgas.png', dpi=300)
    plt.show()
    plt.close()

    return


make_single_image()
