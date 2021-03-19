import os
import sys
import h5py
import numpy as np
import swiftsimio as sw
import velociraptor as vr
from numba import jit
import unyt

sys.path.append("../zooms")

from register import zooms_register, Zoom, Tcut_halogas, calibration_zooms


class Interpolate(object):

    def init(self):
        pass

    def load_table(self):
        self.table = h5py.File(os.path.join(os.path.dirname(__file__), 'X_Ray_table.hdf5'), 'r')
        self.X_Ray = self.table['0.5-2.0keV']['emissivities'][()]
        self.He_bins = self.table['/Bins/He_bins'][()]
        self.missing_elements = self.table['/Bins/Missing_element'][()]

        self.density_bins = self.table['/Bins/Density_bins/'][()]
        self.temperature_bins = self.table['/Bins/Temperature_bins/'][()]
        self.dn = 0.2
        self.dT = 0.1

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
        # mask = np.abs(bins - subdata[i]) < dbins
        # idx_p[i, :] = np.sort(np.argsort(mask)[-2:])
        idx_p[i, :] = np.sort(np.argsort(np.abs(bins - subdata[i]))[:2])

    return idx_p


@jit(nopython=True)
def find_idx_he(subdata, bins):
    idx_p = np.zeros((len(subdata), 2))
    for i in range(len(subdata)):
        idx_p[i, :] = np.sort(np.argsort(np.abs(bins - subdata[i]))[:2])

    return idx_p


@jit(nopython=True)
def find_dx_he(subdata, bins, idx_0):
    dx_p = np.zeros(len(subdata))
    for i in range(len(subdata)):
        dx_p[i] = np.abs(subdata[i] - bins[idx_0[i]]) / (bins[idx_0[i] + 1] - bins[idx_0[i]])
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
        f_n_T0 = (dn - dx_n[i]) / dn * f_00 + dx_n[i] / dn * f_01
        f_n_T1 = (dn - dx_n[i]) / dn * f_10 + dx_n[i] / dn * f_11

        # interpolate temperature
        f_n_T = (dT - dx_T[i]) / dT * f_n_T0 + dx_T[i] / dT * f_n_T1

        # Apply linear scaling for removed metals
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

    interp = Interpolate()
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

    # Calculate the abundance wrt to solar
    abundances = mass_fraction / np.array(element_masses)
    abundance_to_solar = 1 - abundances / 10 ** interp.solar_metallicity

    # Add columns for Calcium and Sulphur and add Iron at the end
    abundance_to_solar = np.c_[
        abundance_to_solar[:, :-1],
        abundance_to_solar[:, -2],
        abundance_to_solar[:, -2],
        abundance_to_solar[:, -1]
    ]

    # Find helium offsets
    idx_he = find_idx_he(np.log10(abundances[:, 1]), interp.He_bins)
    dx_he = find_dx_he(np.log10(abundances[:, 1]), interp.He_bins, idx_he[:, 0].astype(int))

    print('Start interpolation')
    emissivities = get_table_interp(interp.dn, interp.dT, dx_T, dx_n, idx_T.astype(int), idx_n.astype(int),
                                    idx_he.astype(int), dx_he, interp.X_Ray, abundance_to_solar[:, 2:])

    return emissivities


def get_xray_luminosity(
        path_to_snap: str,
        path_to_catalogue: str,
        core_excised: bool = False
) -> unyt.unyt_quantity:

    # Read in halo properties
    vr_catalogue_handle = vr.load(path_to_catalogue)
    a = vr_catalogue_handle.a
    R500c = vr_catalogue_handle.spherical_overdensities.r_500_rhocrit[1].to('Mpc')
    XPotMin = vr_catalogue_handle.positions.xcminpot[1].to('Mpc')
    YPotMin = vr_catalogue_handle.positions.xcminpot[1].to('Mpc')
    ZPotMin = vr_catalogue_handle.positions.xcminpot[1].to('Mpc')

    # Apply spatial mask to particles. SWIFTsimIO needs comoving coordinates
    # to filter particle coordinates, while VR outputs are in physical units.
    # Convert the region bounds to comoving, but keep the CoP and Rcrit in
    # physical units for later use.
    mask = sw.mask(path_to_snap, spatial_only=True)
    region = [
        [XPotMin / a - 0.5 * R500c / a, XPotMin / a + 0.5 * R500c / a],
        [YPotMin / a - 0.5 * R500c / a, YPotMin / a + 0.5 * R500c / a],
        [ZPotMin / a - 0.5 * R500c / a, ZPotMin / a + 0.5 * R500c / a]
    ]
    mask.constrain_spatial(region)
    data = sw.load(path_to_snap, mask=mask)

    # Convert datasets to physical quantities
    # R500c is already in physical units
    data.gas.coordinates.convert_to_physical()
    data.gas.masses.convert_to_physical()
    data.gas.temperatures.convert_to_physical()
    data.gas.densities.convert_to_physical()
    print(data.gas.coordinates)
    # Select hot gas within sphere and without core
    tempGas = data.gas.temperatures
    deltaX = data.gas.coordinates[:, 0] - XPotMin
    deltaY = data.gas.coordinates[:, 1] - YPotMin
    deltaZ = data.gas.coordinates[:, 2] - ZPotMin
    deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) / R500c

    # Keep only particles inside R500crit
    if core_excised:
        index = np.where((deltaR > 0.15) & (deltaR < 1) & (tempGas > 1e5))[0]
    else:
        index = np.where((deltaR < 1) & (tempGas > 1e5))[0]

    del tempGas, deltaX, deltaY, deltaZ, deltaR

    # Compute hydrogen number density and the log10
    # of the temperature to provide to the xray interpolator.
    data_nH = np.log10(data.gas.element_mass_fractions.hydrogen * data.gas.densities.to('g*cm**-3') / unyt.mp)
    data_T = np.log10(data.gas.temperatures.value)

    # Interpolate the Cloudy table to get emissivities
    emissivities = interpolate_X_Ray(
        data_nH,
        data_T,
        data.gas.element_mass_fractions
    )
    emissivities = unyt.unyt_array(10 ** emissivities, 'erg/s/cm**3')
    emissivities = emissivities.to('erg/s/Mpc**3')
    print(emissivities)

    # Compute X-ray luminosities
    # LX = emissivity * gas_mass / gas_density
    xray_luminosities = emissivities * data.gas.masses / data.gas.densities
    xray_luminosities[~np.isfinite(xray_luminosities)] = 0
    print(xray_luminosities)

    return xray_luminosities[index].sum()


if __name__ == '__main__':
    zoom = zooms_register[0]
    print(zoom.run_name)
    LX = get_xray_luminosity(
        path_to_snap=zoom.get_redshift().snapshot_path,
        path_to_catalogue=zoom.get_redshift().catalogue_properties_path
    )
    print(f"X-ray Luminosity: {LX:.3E}")
