import os
import sys
import numpy as np
import unyt
import h5py
import swiftsimio as sw
from scipy.io.idl import readsav

sys.path.append("../zooms")

from register import zooms_register, Zoom, Tcut_halogas, name_list

np.seterr(divide='ignore')
np.seterr(invalid='ignore')


def locate(A, x):
    fdx = A.searchsorted(x)
    fdx = np.clip(fdx, 1, len(A) - 1)
    lt = A[fdx - 1]
    rt = A[fdx]
    fdx -= x - lt < rt - x
    return fdx


def interpolate_xray(data):
    APEC = readsav('/cosma/home/dp004/dc-alta2/xl-zooms/xray/APEC_0.5_2.0keV_interp.idl')

    inde = 0  # 0 - erg/s, 1 - photons
    indz = locate(APEC['redshift'], data.metadata.z)
    indT = locate(APEC["ltemp"], np.log10(data.gas.temperatures))

    ne_nH = np.zeros(len(data.gas.masses.value)) + 1
    ni_nH = np.zeros(len(data.gas.masses.value)) + 1
    mu = np.zeros(len(data.gas.masses.value))
    Lambda = np.zeros(len(data.gas.masses.value), dtype=np.float64)

    # --- Sum element contributions
    # Hydrogen
    H = data.gas.element_mass_fractions.hydrogen.value
    mu += 1.0 / (1.0 + 1.0)
    lN_H_AG = 12.00
    Lambda += APEC["Lambda_hydrogen"][indz, indT, inde]
    # Helium
    He_H = data.gas.element_mass_fractions.helium.value / H
    ne_nH += (He_H) * (1.00794 / 4.002602) * (2.0 / 1.0)
    ni_nH += (He_H) * (1.00794 / 4.002602)
    mu += (He_H) / (1.0 + 2.0)
    AG_He = 10.99 - lN_H_AG
    He_H = 10.0 ** (np.log10(He_H * (1.00794 / 4.002602)) - AG_He)
    Lambda += He_H * APEC["Lambda_helium"][indz, indT, inde]
    del He_H
    # Carbon
    C_H = data.gas.element_mass_fractions.carbon.value / H
    ne_nH += (C_H) * (1.00794 / 12.0107) * (6.0 / 1.0)
    ni_nH += (C_H) * (1.00794 / 12.0107)
    mu += (C_H) / (1.0 + 6.0)
    AG_C = 8.56 - lN_H_AG
    C_H = 10.0 ** (np.log10(C_H * (1.00794 / 12.0107)) - AG_C)
    C_H = 10.0 ** (np.log10(C_H * (1.00794 / 12.0107)) - AG_C)
    Lambda += C_H * APEC["Lambda_carbon"][indz, indT, inde]
    # Nitrogen
    N_H = data.gas.element_mass_fractions.nitrogen.value / H
    ne_nH += (N_H) * (1.00794 / 14.0067) * (7.0 / 1.0)
    ni_nH += (N_H) * (1.00794 / 14.0067)
    mu += (N_H) / (1.0 + 7.0)
    AG_N = 8.05 - lN_H_AG
    N_H = 10.0 ** (np.log10(N_H * (1.00794 / 14.0067)) - AG_N)
    Lambda += N_H * APEC["Lambda_nitrogen"][indz, indT, inde]
    del N_H
    # Oxygen
    O_H = data.gas.element_mass_fractions.oxygen.value / H
    ne_nH += (O_H) * (1.00794 / 15.9994) * (8.0 / 1.0)
    ni_nH += (O_H) * (1.00794 / 15.9994)
    mu += (O_H) / (1.0 + 8.0)
    AG_O = 8.83 - lN_H_AG
    O_H = 10.0 ** (np.log10(O_H * (1.00794 / 15.9994)) - AG_O)
    Lambda += O_H * APEC["Lambda_oxygen"][indz, indT, inde]
    del O_H
    # Neon
    Ne_H = data.gas.element_mass_fractions.neon.value / H
    ne_nH += (Ne_H) * (1.00794 / 20.1797) * (10.0 / 1.0)
    ni_nH += (Ne_H) * (1.00794 / 20.1797)
    mu += (Ne_H) / (1.0 + 10.0)
    AG_Ne = 8.09 - lN_H_AG
    Ne_H = 10.0 ** (np.log10(Ne_H * (1.00794 / 20.1797)) - AG_Ne)
    Lambda += Ne_H * APEC["Lambda_neon"][indz, indT, inde]
    del Ne_H
    # Magnesium
    Mg_H = data.gas.element_mass_fractions.magnesium.value / H
    ne_nH += (Mg_H) * (1.00794 / 24.3050) * (12.0 / 1.0)
    ni_nH += (Mg_H) * (1.00794 / 24.3050)
    mu += (Mg_H) / (1.0 + 12.0)
    AG_Mg = 7.58 - lN_H_AG
    Mg_H = 10.0 ** (np.log10(Mg_H * (1.00794 / 24.3050)) - AG_Mg)
    Lambda += Mg_H * APEC["Lambda_magnesium"][indz, indT, inde]
    del Mg_H
    # Silicon, Sulphur & Calcium
    Si_H = data.gas.element_mass_fractions.silicon.value / H
    Ca_Si = 0.0941736
    S_Si = 0.6054160
    ne_nH += (Si_H) * (1.00794 / 28.0855) * (14.0 / 1.0)
    ne_nH += (Si_H * Ca_Si) * (1.00794 / 40.078) * (20.0 / 1.0)
    ne_nH += (Si_H * S_Si) * (1.00794 / 32.065) * (16.0 / 1.0)
    ni_nH += (Si_H) * (1.00794 / 28.0855)
    ni_nH += (Si_H * Ca_Si) * (1.00794 / 40.078)
    ni_nH += (Si_H * S_Si) * (1.00794 / 32.065)
    mu += (Si_H) / (1.0 + 14.0)
    mu += (Si_H * Ca_Si) / (1.0 + 20.0)
    mu += (Si_H * S_Si) / (1.0 + 16.0)
    AG_Si = 7.55 - lN_H_AG
    AG_Ca = 6.36 - lN_H_AG
    AG_S = 7.21 - lN_H_AG
    Ca_H = 10.0 ** (np.log10((Ca_Si * Si_H) * (1.00794 / 40.078)) - AG_Ca)
    S_H = 10.0 ** (np.log10((S_Si * Si_H) * (1.00794 / 32.065)) - AG_S)
    Si_H = 10.0 ** (np.log10(Si_H * (1.00794 / 28.0855)) - AG_Si)
    Lambda += Si_H * APEC["Lambda_silicon"][indz, indT, inde]
    Lambda += Ca_H * APEC["Lambda_calcium"][indz, indT, inde]
    Lambda += S_H * APEC["Lambda_sulphur"][indz, indT, inde]
    del Si_H, Ca_H, S_H
    # Iron
    Fe_H = data.gas.element_mass_fractions.iron.value / H
    ne_nH += (Fe_H) * (1.00794 / 55.845) * (26.0 / 1.0)
    ni_nH += (Fe_H) * (1.00794 / 55.845)
    mu += (Fe_H) / (1.0 + 26.0)
    AG_Fe = 7.67 - lN_H_AG
    Fe_H = 10.0 ** (np.log10(Fe_H * (1.00794 / 55.845)) - AG_Fe)
    Lambda += Fe_H * APEC["Lambda_iron"][indz, indT, inde]
    del H, Fe_H, indT

    Lambda = unyt.unyt_array(Lambda, 'erg/s*cm**3')
    erg2keV = 1.60215e-9  # erg --> keV conversion factor

    # --- Calculate observables
    pix = 1
    Lx = Lambda * (data.gas.densities * (ne_nH / ((ne_nH + ni_nH) * mu * unyt.mp)) ** 2.0) * data.gas.masses / ne_nH
    Sx = Lx / (4.0 * np.pi * pix * pix) / ((180.0 * 60.0 / np.pi) ** 2)
    Ypix = (unyt.thompson_cross_section_cgs.v / (511.0 * erg2keV)) * unyt.kb_cgs * data.gas.temperatures * (
            data.gas.masses / (mu * unyt.mp_cgs.v)) * (ne_nH / (ne_nH + ni_nH)) / (pix * pix)

    return Lx, Sx, Ypix


def process_single_halo(
        path_to_snap: str,
        path_to_catalogue: str
):
    # Read in halo properties
    with h5py.File(f'{path_to_catalogue}', 'r') as h5file:
        M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
        R500c = unyt.unyt_quantity(h5file['/SO_R_500_rhocrit'][0], unyt.Mpc)
        XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc)
        YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc)
        ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc)

    # Read in gas particles to compute the core-excised temperature
    mask = sw.mask(f'{path_to_snap}', spatial_only=False)
    region = [[XPotMin - 0.5 * R500c, XPotMin + 0.5 * R500c],
              [YPotMin - 0.5 * R500c, YPotMin + 0.5 * R500c],
              [ZPotMin - 0.5 * R500c, ZPotMin + 0.5 * R500c]]
    mask.constrain_spatial(region)
    mask.constrain_mask(
        "gas", "temperatures",
        1.e5 * mask.units.temperature,
        1.e10 * mask.units.temperature
    )

    data = sw.load(f'{path_to_snap}', mask=mask)

    # Select hot gas within sphere and without core
    deltaX = data.gas.coordinates[:, 0] - XPotMin
    deltaY = data.gas.coordinates[:, 1] - YPotMin
    deltaZ = data.gas.coordinates[:, 2] - ZPotMin
    deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2)

    # Keep only particles inside 5 R500crit
    index = np.where((deltaR > 0.15 * R500c) & (deltaR < R500c))[0]
    data.gas.radial_distances = deltaR[index]
    data.gas.densities = data.gas.densities[index]
    data.gas.masses = data.gas.masses[index]
    data.gas.temperatures = data.gas.temperatures[index]

    data.gas.element_mass_fractions.hydrogen = data.gas.element_mass_fractions.hydrogen[index]
    data.gas.element_mass_fractions.helium = data.gas.element_mass_fractions.helium[index]
    data.gas.element_mass_fractions.carbon = data.gas.element_mass_fractions.carbon[index]
    data.gas.element_mass_fractions.nitrogen = data.gas.element_mass_fractions.nitrogen[index]
    data.gas.element_mass_fractions.oxygen = data.gas.element_mass_fractions.oxygen[index]
    data.gas.element_mass_fractions.neon = data.gas.element_mass_fractions.neon[index]
    data.gas.element_mass_fractions.magnesium = data.gas.element_mass_fractions.magnesium[index]
    data.gas.element_mass_fractions.silicon = data.gas.element_mass_fractions.silicon[index]
    data.gas.element_mass_fractions.iron = data.gas.element_mass_fractions.iron[index]

    Lx, Sx, Ypix = interpolate_xray(data)

    print(f"M_500_crit = {M500c:.3E}")
    print(f'LX = {np.sum(Lx):.3E}')

    return np.sum(Lx), Sx, Ypix


if __name__ == '__main__':
    process_single_halo(
        path_to_snap=zooms_register[0].snapshot_file,
        path_to_catalogue=zooms_register[0].catalog_file
    )
