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
from convergence_radius import convergence_radius

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--keywords', type=str, nargs='+', required=True)
parser.add_argument('-e', '--observ-errorbars', default=False, required=False, action='store_true')
parser.add_argument('-r', '--redshift-index', type=int, default=36, required=False,
                    choices=list(range(len(calibration_zooms.get_snap_redshifts()))))
parser.add_argument('-m', '--mass-estimator', type=str.lower, default='crit', required=True,
                    choices=['crit', 'true', 'hse'])
parser.add_argument('-q', '--quiet', default=False, required=False, action='store_true')
args = parser.parse_args()

# Constants
mean_molecular_weight = 0.59
mean_atomic_weight_per_free_electron = 1.14
bins = 40
radius_bounds = [0.01, 2.5]  # In units of R500crit
sampling_method = 'shell_density'
sampling_method = 'no_binning'
# sampling_method = 'particle_density'
field_name = 'entropy'


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def histogram_unyt(
        data: unyt.unyt_array,
        bins: unyt.unyt_array = None,
        weights: unyt.unyt_array = None,
        **kwargs,
) -> Tuple[unyt.unyt_array]:
    assert data.shape == weights.shape, (
        "Data and weights arrays must have the same shape. "
        f"Detected data {data.shape}, weights {weights.shape}."
    )

    assert data.units == bins.units, (
        "Data and bins must have the same units. "
        f"Detected data {data.units}, bins {bins.units}."
    )

    hist, bin_edges = np.histogram(data.value, bins=bins.value, weights=weights.value, **kwargs)
    hist *= weights.units
    bin_edges *= data.units

    return hist, bin_edges


def cumsum_unyt(data: unyt.unyt_array, **kwargs) -> unyt.unyt_array:
    res = np.cumsum(data.value, **kwargs)

    return res * data.units


def profile_3d_single_halo(
        path_to_snap: str,
        path_to_catalogue: str,
        weights: str,
        hse_dataset: pd.Series = None,
) -> tuple:
    # Read in halo properties
    with h5.File(path_to_catalogue, 'r') as h5file:
        scale_factor = float(h5file['/SimulationInfo'].attrs['ScaleFactor'])
        M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
        R500c = unyt.unyt_quantity(h5file['/SO_R_500_rhocrit'][0], unyt.Mpc) / scale_factor
        XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc) / scale_factor
        YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc) / scale_factor
        ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc) / scale_factor

        # If no custom aperture, select R500c as default
        if hse_dataset is not None:
            assert R500c.units == hse_dataset["R500hse"].units
            assert M500c.units == hse_dataset["M500hse"].units
            R500c = hse_dataset["R500hse"]
            M500c = hse_dataset["M500hse"]

    # Read in gas particles
    mask = sw.mask(path_to_snap, spatial_only=False)
    region = [[XPotMin - radius_bounds[1] * R500c, XPotMin + radius_bounds[1] * R500c],
              [YPotMin - radius_bounds[1] * R500c, YPotMin + radius_bounds[1] * R500c],
              [ZPotMin - radius_bounds[1] * R500c, ZPotMin + radius_bounds[1] * R500c]]
    mask.constrain_spatial(region)
    mask.constrain_mask("gas", "temperatures", Tcut_halogas * mask.units.temperature, 1.e12 * mask.units.temperature)
    data = sw.load(path_to_snap, mask=mask)

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

    # Select hot gas within sphere
    posGas = data.gas.coordinates
    deltaX = posGas[:, 0] - XPotMin
    deltaY = posGas[:, 1] - YPotMin
    deltaZ = posGas[:, 2] - ZPotMin
    deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2)

    # Calculate particle mass and rho_crit
    unitLength = data.metadata.units.length
    unitMass = data.metadata.units.mass

    rho_crit = unyt.unyt_quantity(
        data.metadata.cosmology_raw['Critical density [internal units]'] / scale_factor ** 3,
        unitMass / unitLength ** 3
    )
    dm_masses = data.dark_matter.masses.to('Msun')
    zoom_mass_resolution = dm_masses[0]

    # Since useful for different applications, attach datasets
    data.gas.mass_weighted_temperatures = data.gas.masses * data.gas.temperatures

    # Rescale profiles to R500c
    radial_distance = deltaR / R500c
    assert radial_distance.units == unyt.dimensionless

    # Compute convergence radius
    conv_radius = convergence_radius(deltaR, data.gas.masses.to('Msun'), rho_crit.to('Msun/Mpc**3')) / R500c

    # Construct bins for mass-weighted quantities and retrieve bin_edges
    lbins = np.logspace(np.log10(radius_bounds[0]), np.log10(radius_bounds[1]), bins) * radial_distance.units
    mass_weights, bin_edges = histogram_unyt(radial_distance, bins=lbins, weights=data.gas.masses)

    # Replace zeros with Nans
    mass_weights[mass_weights == 0] = np.nan

    bin_centre = np.sqrt(bin_edges[1:] * bin_edges[:-1])

    # Allocate weights
    if weights.lower() == 'gas_mass':
        hist = mass_weights / M500c.to(mass_weights.units)

        ylabel = r'$M(dR) / M_{500{\rm crit}}$'

    elif weights.lower() == 'gas_mass_cumulative':
        hist = cumsum_unyt(mass_weights) / M500c.to(mass_weights.units)

        ylabel = r'$M(<R) / M_{500{\rm crit}}$'

    elif weights.lower() == 'gas_density':
        hist, _ = histogram_unyt(radial_distance, bins=lbins, weights=data.gas.densities)
        hist /= rho_crit.to(hist.units)
        hist *= bin_centre ** 2

        ylabel = r'$(\rho_{\rm gas}/\rho_{\rm crit})\ (R/R_{500{\rm crit}})^3 $'

    elif weights.lower() == 'mass_weighted_temps':
        weights_field = data.gas.mass_weighted_temperatures
        hist, _ = histogram_unyt(radial_distance, bins=lbins, weights=weights_field)
        hist /= mass_weights

        ylabel = r'$T$ [K]'

    elif weights.lower() == 'mass_weighted_temps_kev':
        weights_field = data.gas.mass_weighted_temperatures
        hist, _ = histogram_unyt(radial_distance, bins=lbins, weights=weights_field)
        hist /= mass_weights
        hist = (hist * unyt.boltzmann_constant).to('keV')

        # Make dimensionless, divide by (k_B T_500crit)
        # unyt.G.in_units('Mpc*(km/s)**2/(1e10*Msun)')
        norm = unyt.G * mean_molecular_weight * M500c * unyt.mass_proton / 2 / R500c
        norm = norm.to('keV')
        hist /= norm

        ylabel = r'$(k_B T/k_B T_{500{\rm crit}})$'

    elif weights.lower() == 'entropy':

        if sampling_method.lower() == 'shell_density':

            volume_shell = (4. * np.pi / 3.) * (R500c ** 3) * ((bin_edges[1:]) ** 3 - (bin_edges[:-1]) ** 3)
            density_gas = mass_weights / volume_shell
            mean_density_R500c = (3 * M500c * obs.cosmic_fbary / (4 * np.pi * R500c ** 3)).to(density_gas.units)

            kBT, _ = histogram_unyt(radial_distance, bins=lbins, weights=data.gas.mass_weighted_temperatures)
            kBT *= unyt.boltzmann_constant
            kBT /= mass_weights
            kBT = kBT.to('keV')
            kBT_500crit = unyt.G * mean_molecular_weight * M500c * unyt.mass_proton / 2 / R500c
            kBT_500crit = kBT_500crit.to(kBT.units)

            # Note: the ratio of densities is the same as ratio of electron number densities
            hist = kBT / kBT_500crit * (mean_density_R500c / density_gas) ** (2 / 3)

        elif sampling_method.lower() == 'particle_density':

            n_e = data.gas.densities
            ne_500crit = 3 * M500c * obs.cosmic_fbary / (4 * np.pi * R500c ** 3)

            kBT = unyt.boltzmann_constant * data.gas.mass_weighted_temperatures
            kBT_500crit = unyt.G * mean_molecular_weight * M500c * unyt.mass_proton / 2 / R500c

            weights_field = kBT / kBT_500crit * (ne_500crit / n_e) ** (2 / 3)
            hist, _ = histogram_unyt(radial_distance, bins=lbins, weights=weights_field)
            hist /= mass_weights

        elif sampling_method.lower() == 'no_binning':

            n_e = data.gas.densities
            ne_500crit = 3 * M500c * obs.cosmic_fbary / (4 * np.pi * R500c ** 3)
            kBT = unyt.boltzmann_constant * data.gas.temperatures
            kBT_500crit = unyt.G * mean_molecular_weight * M500c * unyt.mass_proton / 2 / R500c
            weights_field = kBT / kBT_500crit * (ne_500crit / n_e) ** (2 / 3)

            bin_centre = radial_distance
            hist = weights_field

        ylabel = r'$K/K_{500{\rm crit}}$'

    elif weights.lower() == 'entropy_physical':

        if sampling_method.lower() == 'shell_density':

            volume_shell = (4. * np.pi / 3.) * (R500c ** 3) * ((bin_edges[1:]) ** 3 - (bin_edges[:-1]) ** 3)
            density_gas = mass_weights / volume_shell
            number_density_gas = density_gas / (mean_molecular_weight * unyt.mass_proton)
            number_density_gas = number_density_gas.to('1/cm**3')

            kBT, _ = histogram_unyt(radial_distance, bins=lbins, weights=data.gas.mass_weighted_temperatures)
            kBT *= unyt.boltzmann_constant
            kBT /= mass_weights
            kBT = kBT.to('keV')

            # Note: the ratio of densities is the same as ratio of electron number densities
            hist = kBT / number_density_gas ** (2 / 3)
            hist = hist.to('keV*cm**2')

        elif sampling_method.lower() == 'particle_density':

            number_density_gas = data.gas.densities / (mean_molecular_weight * unyt.mass_proton)
            number_density_gas = number_density_gas.to('1/cm**3')

            kBT = unyt.boltzmann_constant * data.gas.mass_weighted_temperatures

            weights_field = kBT / number_density_gas ** (2 / 3)
            hist, _ = histogram_unyt(radial_distance, bins=lbins, weights=weights_field)
            hist /= mass_weights
            hist = hist.to('keV*cm**2')

        elif sampling_method.lower() == 'no_binning':

            number_density_gas = data.gas.densities / (mean_molecular_weight * unyt.mass_proton)
            number_density_gas = number_density_gas.to('1/cm**3')
            kBT = unyt.boltzmann_constant * data.gas.temperatures
            weights_field = kBT / number_density_gas ** (2 / 3)

            bin_centre = radial_distance
            hist = weights_field

        ylabel = r'$K$   [keV cm$^2$]'

    elif weights.lower() == 'pressure':

        if sampling_method.lower() == 'shell_density':

            volume_shell = (4. * np.pi / 3.) * (R500c ** 3) * ((bin_edges[1:]) ** 3 - (bin_edges[:-1]) ** 3)
            density_gas = mass_weights / volume_shell
            number_density_gas = density_gas / (mean_molecular_weight * unyt.mass_proton)
            number_density_gas = number_density_gas.to('1/cm**3')

            kBT, _ = histogram_unyt(radial_distance, bins=lbins, weights=data.gas.mass_weighted_temperatures)
            kBT *= unyt.boltzmann_constant
            kBT /= mass_weights
            kBT = kBT.to('keV')

            # Note: the ratio of densities is the same as ratio of electron number densities
            hist = kBT * number_density_gas
            hist = hist.to('keV/cm**3')

        elif sampling_method.lower() == 'particle_density':

            weights_field = data.gas.pressures * data.gas.masses
            hist, _ = histogram_unyt(radial_distance, bins=lbins, weights=weights_field)
            hist /= mass_weights

        # Make dimensionless, divide by P_500crit
        norm = 500 * obs.cosmic_fbary * rho_crit * unyt.G * M500c / 2 / R500c
        hist /= norm.to(hist.units)
        hist *= bin_centre ** 3

        ylabel = r'$(P/P_{500{\rm crit}})\ (R/R_{500{\rm crit}})^3 $'

    else:
        raise ValueError(f"Unrecognized weighting field: {weights}.")

    return bin_centre, hist, ylabel, conv_radius, M500c


@utils.set_scaling_relation_name(os.path.splitext(os.path.basename(__file__))[0])
@utils.set_output_names([
    'bin_centre',
    field_name,
    'ylabel',
    'convergence_radius',
    'M500'
])
def _process_single_halo(zoom: Zoom):
    # Select redshift
    snapshot_file = zoom.get_redshift(args.redshift_index).snapshot_path
    catalog_file = zoom.get_redshift(args.redshift_index).catalogue_properties_path

    if args.mass_estimator == 'crit' or args.mass_estimator == 'true':

        return profile_3d_single_halo(snapshot_file, catalog_file, weights=field_name)

    elif args.mass_estimator == 'hse':
        try:
            hse_catalogue = pd.read_pickle(f'{calibration_zooms.output_directory}/hse_massbias.pkl')
        except FileExistsError as error:
            raise FileExistsError(
                f"{error}\nPlease, consider first generating the HSE catalogue for better performance."
            )

        hse_catalogue_names = hse_catalogue['Run name'].values.tolist()
        print(f"Looking for HSE results in the catalogue - zoom: {zoom.run_name}")
        if zoom.run_name in hse_catalogue_names:
            i = hse_catalogue_names.index(zoom.run_name)
            hse_entry = hse_catalogue.loc[i]
        else:
            raise ValueError(f"{zoom.run_name} not found in HSE catalogue. Please, regenerate the catalogue.")

        return profile_3d_single_halo(snapshot_file, catalog_file, weights=field_name, hse_dataset=hse_entry)


def plot_profiles(results: pd.DataFrame):
    fig, ax = plt.subplots()
    legend_handles = []
    for i in range(len(results)):

        run_style = style.get_style_for_object(results.loc[i, "Run name"])
        if run_style['Legend handle'] not in legend_handles:
            legend_handles.append(run_style['Legend handle'])

        # Plot only profiles outside convergence radius
        convergence_index = np.where(results['bin_centre'][i] > results['convergence_radius'][i])[0]

        ax.plot(
            results['bin_centre'][i][convergence_index],
            results[field_name][i][convergence_index],
            linestyle=run_style['Line style'],
            color=run_style['Color'],
            linewidth=0.5,
            alpha=0.6,
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$R/R_{500{\rm crit}}$')
    ax.set_ylabel(results['ylabel'][0])
    if not args.quiet:
        plt.show()
    plt.close()


if __name__ == "__main__":
    results = utils.process_catalogue(_process_single_halo, find_keyword=args.keywords)
    plot_profiles(results)
