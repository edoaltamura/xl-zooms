import os
import sys
import numpy as np
import unyt
import h5py
import swiftsimio as sw
from numba import jit

sys.path.append("../zooms")

from register import zooms_register, Zoom, Tcut_halogas, name_list

np.seterr(divide='ignore')
np.seterr(invalid='ignore')


class interpolate:
    def __init__(self):
        pass

    def load_table(self):
        self.dn = 0.2
        self.dT = 0.1

        self.table = h5py.File('/cosma/home/dp004/dc-alta2/xl-zooms/xray/X_Ray_table.hdf5', 'r')
        self.X_Ray = self.table['0.5-2.0keV']['emissivities'][()]
        self.He_bins = self.table['/Bins/He_bins'][()]
        self.missing_elements = self.table['/Bins/Missing_element'][()]
        self.density_bins = self.table['/Bins/Density_bins/'][()]
        self.temperature_bins = self.table['/Bins/Temperature_bins/'][()]
        self.solar_metallicity = self.table['/Bins/Solar_metallicities/'][()]


@jit(nopython=True)
def find_dx(subdata, bins, idx_0):
    dx_p = np.zeros(len(subdata))
    for i in range(len(subdata)):
        dx_p[i] = np.abs(bins[idx_0[i]] - subdata[i])

    return dx_p


@jit(nopython=True)
def find_idx(subdata, bins, dbins):
    idx_p = np.zeros((len(subdata), 2))
    for i in range(len(subdata)):
        mask = np.abs(bins - subdata[i]) < dbins
        idx_p[i, :] = np.sort(np.argsort(mask)[-2:])

    return idx_p


@jit(nopython=True)
def find_idx_he(subdata, bins):
    idx_p = np.zeros((len(subdata), 2))
    for i in range(len(subdata)):
        idx_p[i, :] = np.sort(np.argsort(np.abs(bins - subdata[i]))[-2:])

    return idx_p


@jit(nopython=True)
def find_dx_he(subdata, bins, idx_0):
    dx_p = np.zeros(len(subdata))
    for i in range(len(subdata)):
        dx_p[i] = np.abs(subdata[i] - bins[idx_0[i]]) / (bins[idx_0[i + 1]] - bins[idx_0[i]])
    # dx_p1[i] = np.abs(bins[idx_0[i+1]] - subdata[i])

    return dx_p


@jit(nopython=True)
def get_table_interp(dn, dT, dx_T, dx_n, idx_T, idx_n, idx_he, dx_he, X_Ray, abundance_to_solar):
    f_n_T_Z = np.zeros(len(idx_n[:, 0]))
    for i in range(len(idx_n[:, 0])):
        # interpolate He
        f_000 = X_Ray[0, idx_he[i, 0], :, idx_T[i, 0], idx_n[i, 0]]
        f_001 = X_Ray[0, idx_he[i, 0], :, idx_T[i, 0], idx_n[i, 1]]
        f_010 = X_Ray[0, idx_he[i, 0], :, idx_T[i, 1], idx_n[i, 0]]
        f_011 = X_Ray[0, idx_he[i, 0], :, idx_T[i, 1], idx_n[i, 1]]

        f_100 = X_Ray[0, idx_he[i, 1], :, idx_T[i, 0], idx_n[i, 0]]
        f_101 = X_Ray[0, idx_he[i, 1], :, idx_T[i, 0], idx_n[i, 1]]
        f_110 = X_Ray[0, idx_he[i, 1], :, idx_T[i, 1], idx_n[i, 0]]
        f_111 = X_Ray[0, idx_he[i, 1], :, idx_T[i, 1], idx_n[i, 1]]

        f_00 = f_000 * (1 - dx_he[i]) + f_100 * dx_he[i]
        f_01 = f_001 * (1 - dx_he[i]) + f_101 * dx_he[i]
        f_10 = f_010 * (1 - dx_he[i]) + f_110 * dx_he[i]
        f_11 = f_011 * (1 - dx_he[i]) + f_111 * dx_he[i]

        # interpolate density
        f_n_T0 = (dn - dx_n[i]) / dn * f_00 + dx_n[i] / dn * f_10
        f_n_T1 = (dn - dx_n[i]) / dn * f_01 + dx_n[i] / dn * f_11

        # interpolate temperature
        f_n_T = (dT - dx_T[i]) / dT * f_n_T0 + dx_T[i] / dT * f_n_T1

        f_n_T_Z_temp = f_n_T[-1]
        for j in range(len(f_n_T) - 1):
            f_n_T_Z_temp -= (f_n_T[-1] - f_n_T[j]) * abundance_to_solar[i, j]

        f_n_T_Z[i] = f_n_T_Z_temp

    return f_n_T_Z


def interpolate_X_Ray(data_n, data_T, element_mass_fractions):
    mass_fraction = np.zeros((len(data_n), 9))

    # get individual mass fraction
    mass_fraction[:, 0] = element_mass_fractions.hydrogen
    mass_fraction[:, 1] = element_mass_fractions.helium
    mass_fraction[:, 2] = element_mass_fractions.carbon
    mass_fraction[:, 3] = element_mass_fractions.nitrogen
    mass_fraction[:, 4] = element_mass_fractions.oxygen
    mass_fraction[:, 5] = element_mass_fractions.neon
    mass_fraction[:, 6] = element_mass_fractions.magnesium
    mass_fraction[:, 7] = element_mass_fractions.silicon
    mass_fraction[:, 8] = element_mass_fractions.iron

    interp = interpolate()
    interp.load_table()

    # Find density offsets
    idx_n = find_idx(data_n, interp.density_bins, interp.dn)
    dx_n = find_dx(data_n, interp.density_bins, idx_n[:, 0].astype(int))

    # Find temperature offsets
    idx_T = find_idx(data_T, interp.temperature_bins, interp.dT)
    dx_T = find_dx(data_T, interp.temperature_bins, idx_T[:, 0].astype(int))

    # Find element offsets
    # mass of ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'iron']
    element_masses = [1, 4.0026, 12.0107, 14.0067, 15.999, 20.1797, 24.305, 28.0855, 55.845]
    abundances = np.log10(mass_fraction / np.array(element_masses))

    abundance_to_solar = 1 - (abundances / interp.solar_metallicity)

    abundance_to_solar = np.c_[
        abundance_to_solar[:, :-1],
        abundance_to_solar[:, -2],
        abundance_to_solar[:, -2],
        abundance_to_solar[:, -1]
    ]  # Add columns for Calcium and Sulphur and add Iron at the end

    # Find helium offsets
    idx_he = find_idx_he(abundances[:, 1], interp.He_bins)
    dx_he = find_dx(abundances[:, 1], interp.He_bins, idx_he[:, 0].astype(int))

    print('Start interpolation')
    emissivities = get_table_interp(
        interp.dn,
        interp.dT,
        dx_T,
        dx_n,
        idx_T.astype(int),
        idx_n.astype(int),
        idx_he.astype(int),
        dx_he,
        interp.X_Ray,
        abundance_to_solar[:, 2:]
    )

    return emissivities


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

    # Compute number density
    data_n = np.log10((data.gas.densities.in_cgs() / unyt.proton_mass_cgs).value)

    # get temperature
    data_T = np.log10(data.gas.temperatures.value)

    # interpolate
    emissivity = 10 ** interpolate_X_Ray(
        data_n,
        data_T,
        data.gas.element_mass_fractions
    ) * unyt.erg * unyt.cm ** -3 / unyt.s

    # Compute X-ray luminosities
    # xray_luminosities = data.gas.densities / unyt.proton_mass / 0.6 * \
    #                     data.gas.masses / unyt.proton_mass / 0.6 * \
    #                     emissivity
    xray_luminosities = emissivity / (unyt.proton_mass / 0.6) ** 2
    xray_luminosities[~np.isfinite(xray_luminosities)] = 0

    print(f"M_500_crit: {M500c:.3E}")
    print(f"X-luminosity (0.15-1 x R500c): {np.sum(xray_luminosities):.3E}")

    return np.sum(xray_luminosities)


if __name__ == '__main__':

    process_single_halo(
        path_to_snap=zooms_register[0].snapshot_file,
        path_to_catalogue=zooms_register[0].catalog_file
    )