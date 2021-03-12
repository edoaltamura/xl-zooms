import os
import sys
import unyt
import argparse
import h5py as h5
import numpy as np
import pandas as pd
import swiftsimio as sw
from typing import Tuple
import matplotlib.pyplot as plt

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

# Make the register backend visible to the script
sys.path.append("../observational_data")
sys.path.append("../scaling_relations")
sys.path.append("../zooms")

from register import zooms_register, Zoom, Tcut_halogas, calibration_zooms
import observational_data as obs
import scaling_utils as utils
import scaling_style as style

cosmology = obs.Observations().cosmo_model
fbary = cosmology.Ob0 / cosmology.Om0  # Cosmic baryon fraction
mean_molecular_weight = 0.59
mean_atomic_weight_per_free_electron = 1.14

entropy_scaling = 'physical'
entropy_shell_radius = (0.1, 'R500')
entropy_shell_thickness = unyt.unyt_quantity(10, 'kpc')


def process_single_halo(
        path_to_snap: str,
        path_to_catalogue: str,
        hse_dataset: pd.Series = None,
) -> tuple:
    # Read in halo properties
    with h5.File(f'{path_to_catalogue}', 'r') as h5file:
        scale_factor = float(h5file['/SimulationInfo'].attrs['ScaleFactor'])
        XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc) / scale_factor
        YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc) / scale_factor
        ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc) / scale_factor
        M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
        R2500c = unyt.unyt_quantity(h5file['/SO_R_2500_rhocrit'][0], unyt.Mpc) / scale_factor
        R500c = unyt.unyt_quantity(h5file['/SO_R_500_rhocrit'][0], unyt.Mpc) / scale_factor
        R200c = unyt.unyt_quantity(h5file['/R_200crit'][0], unyt.Mpc) / scale_factor

    # If no custom aperture, select R500c as default
    if hse_dataset is not None:
        assert R500c.units == hse_dataset["R500hse"].units
        assert M500c.units == hse_dataset["M500hse"].units
        R500c = hse_dataset["R500hse"]
        M500c = hse_dataset["M500hse"]

    # Read in gas particles
    mask = sw.mask(f'{path_to_snap}', spatial_only=False)
    region = [[XPotMin - R500c, XPotMin + R500c],
              [YPotMin - R500c, YPotMin + R500c],
              [ZPotMin - R500c, ZPotMin + R500c]]
    mask.constrain_spatial(region)
    mask.constrain_mask("gas", "temperatures", Tcut_halogas * mask.units.temperature, 1.e12 * mask.units.temperature)
    data = sw.load(f'{path_to_snap}', mask=mask)

    # Convert datasets to physical quantities
    R500c *= scale_factor
    XPotMin *= scale_factor
    YPotMin *= scale_factor
    ZPotMin *= scale_factor
    data.gas.coordinates.convert_to_physical()
    data.gas.masses.convert_to_physical()
    data.gas.temperatures.convert_to_physical()
    data.gas.densities.convert_to_physical()
    data.gas.pressures.convert_to_physical()
    data.gas.entropies.convert_to_physical()
    data.dark_matter.masses.convert_to_physical()

    posGas = data.gas.coordinates
    massGas = data.gas.masses
    mass_weighted_tempGas = data.gas.temperatures * data.gas.masses

    # Select hot gas within sphere
    deltaX = posGas[:, 0] - XPotMin
    deltaY = posGas[:, 1] - YPotMin
    deltaZ = posGas[:, 2] - ZPotMin
    deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2)
    index = np.where(deltaR < R500c)[0]
    Mhot500c = np.sum(massGas[index])
    fhot500c = Mhot500c / M500c

    # Calculate entropy
    factor, scale_radius = entropy_shell_radius
    if 'r500' in scale_radius.lower():
        entropy_radius = factor * R500c
    elif 'r200' in scale_radius.lower():
        entropy_radius = factor * R200c
    elif 'r2500' in scale_radius.lower():
        entropy_radius = factor * R2500c

    shell_index = np.where(
        (deltaR > entropy_radius - entropy_shell_thickness / 2) &
        (deltaR < entropy_radius + entropy_shell_thickness / 2)
    )[0]
    mass_shell = np.sum(massGas[shell_index])
    volume_shell = (4. * np.pi / 3.) * (
            (entropy_radius + entropy_shell_thickness / 2) ** 3 -
            (entropy_radius - entropy_shell_thickness / 2) ** 3
    )
    density_shell = mass_shell / volume_shell

    kBT_shell = np.sum(mass_weighted_tempGas[shell_index])
    kBT_shell /= mass_shell
    kBT_shell *= unyt.boltzmann_constant
    kBT_shell = kBT_shell.to('keV')

    mean_density_R500c = (3 * M500c * fbary / (4 * np.pi * R500c ** 3)).to(density_shell.units)
    kBT_500crit = unyt.G * mean_molecular_weight * M500c * unyt.mass_proton / 2 / R500c
    kBT_500crit = kBT_500crit.to(kBT_shell.units)

    if entropy_scaling.lower() == 'k500':
        # Note: the ratio of densities is the same as ratio of electron number densities
        entropy = kBT_shell / kBT_500crit * (mean_density_R500c / density_shell) ** (2 / 3)

    elif entropy_scaling.lower() == 'physical':

        number_density_gas = density_shell / (mean_molecular_weight * unyt.mass_proton)
        number_density_gas = number_density_gas.to('1/cm**3')
        entropy = kBT_shell / number_density_gas ** (2 / 3)
        entropy = entropy.to('keV*cm**2')

    return M500c.to(unyt.Solar_Mass), Mhot500c.to(unyt.Solar_Mass), fhot500c, entropy, kBT_500crit

@utils.set_scaling_relation_name(os.path.splitext(os.path.basename(__file__))[0])
@utils.set_output_names([
    'M_500crit',
    'Mhot500c',
    'fhot500c',
    'entropy',
    'kBT_500crit'
])
def _process_single_halo(zoom: Zoom):
    return process_single_halo(zoom.snapshot_file, zoom.catalog_file)


def m_500_entropy(results: pd.DataFrame):
    fig, ax = plt.subplots()
    legend_handles = []
    for i in range(len(results)):

        run_style = style.get_style_for_object(results.loc[i, "Run name"])
        if run_style['Legend handle'] not in legend_handles:
            legend_handles.append(run_style['Legend handle'])

        ax.scatter(
            results.loc[i, "M_500crit"],
            results.loc[i, "entropy"],
            marker=run_style['Marker style'],
            c=run_style['Color'],
            s=run_style['Marker size'],
            alpha=1,
            edgecolors='none',
            zorder=5
        )

    # Build legends
    legend_sims = plt.legend(handles=legend_handles, loc=2)
    ax.add_artist(legend_sims)

    ax.set_xlabel(r'$M_{{500{{\rm crit}}}}$ [${0}$]'.format(
        results.loc[0, "M_500crit"].units.latex_repr
    ))
    ax.set_ylabel(r'Entropy $\ (r={0:.1g}\ R_{{{1:d}{{\rm crit}}}})$ [${2:s}$]'.format(
        entropy_shell_radius[0],
        int(''.join([i for i in entropy_shell_radius[1] if i.isdigit()])),
        results.loc[0, "entropy"].units.latex_repr
    ))
    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.savefig(f'{zooms_register[0].output_directory}/m500_k500.png', dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    import sys

    if sys.argv[1]:
        keyword = sys.argv[1]
    else:
        keyword = 'Ref'

    results = utils.process_catalogue(_process_single_halo, find_keyword=keyword)
    m_500_entropy(results)
