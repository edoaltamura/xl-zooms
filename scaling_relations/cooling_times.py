import os
import h5py as h5
from typing import Tuple
import numpy as np
from unyt import (
    unyt_array,
    unyt_quantity,
    mh, G, mp, K, kb, cm, Solar_Mass
)
from warnings import warn
import scipy.interpolate as sci
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from literature import Cosmology, Sun2009, Pratt2010
from register import Zoom, args, cooling_table, default_output_directory
from scaling_relations import HaloProperty, SODelta500
from hydrostatic_estimates import HydrostaticEstimator

mean_molecular_weight = 0.59
mean_atomic_weight_per_free_electron = 1.14
primordial_hydrogen_mass_fraction = 0.76
solar_metallicity = 0.0133714


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if f < 1e-2 or f > 1e3:
        float_str = "{0:.0e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def histogram_unyt(
        data: unyt_array,
        bins: unyt_array = None,
        weights: unyt_array = None,
        **kwargs,
) -> Tuple[unyt_array]:
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


def draw_adiabats(axes, density_bins, temperature_bins):
    density_interps, temperature_interps = np.meshgrid(density_bins, temperature_bins)
    entropy_interps = temperature_interps * K * kb / (density_interps / cm ** 3) ** (2 / 3)
    entropy_interps = entropy_interps.to('keV*cm**2').value

    # Define entropy levels to plot
    levels = [1e-4, 1e-2, 1, 1e2, 1e4]
    fmt = {value: f'${latex_float(value)}$ keV cm$^2$' for value in levels}
    contours = axes.contour(
        density_interps,
        temperature_interps,
        entropy_interps,
        levels,
        colors='aqua',
        linewidths=0.3,
        alpha=0.5
    )

    # work with logarithms for loglog scale
    # middle of the figure:
    xmin, xmax, ymin, ymax = axes.axis()
    logmid = (np.log10(xmin) + np.log10(xmax)) / 2, (np.log10(ymin) + np.log10(ymax)) / 2

    label_pos = []
    i = 0
    for line in contours.collections:
        for path in line.get_paths():
            logvert = np.log10(path.vertices)

            # Align with same x-value
            if levels[i] > 1:
                log_rho = -4.5
            else:
                log_rho = 15

            # logmid = log_rho, np.log10(levels[i]) - 2 * log_rho / 3
            i += 1

            # find closest point
            logdist = np.linalg.norm(logvert - logmid, ord=2, axis=1)
            min_ind = np.argmin(logdist)
            label_pos.append(10 ** logvert[min_ind, :])

    # Draw contour labels
    axes.clabel(
        contours,
        inline=True,
        inline_spacing=3,
        rightside_up=True,
        colors='aqua',
        fontsize=5,
        fmt=fmt,
        manual=label_pos,
    )


def draw_k500(axes, density_bins, temperature_bins, k500):
    density_interps, temperature_interps = np.meshgrid(density_bins, temperature_bins)
    entropy_interps = temperature_interps * K * kb / (density_interps / cm ** 3) ** (2 / 3)
    entropy_interps = entropy_interps.to('keV*cm**2').value

    # Define entropy levels to plot
    levels = [float(k500.value)]
    fmt = {levels[0]: f'$K_{{500}} = {levels[0]:.1f}$ keV cm$^2$'}
    contours = axes.contour(
        density_interps,
        temperature_interps,
        entropy_interps,
        levels,
        colors='red',
        linewidths=0.3,
        alpha=0.5
    )

    # work with logarithms for loglog scale
    # middle of the figure:
    xmin, xmax, ymin, ymax = axes.axis()
    logmid = (np.log10(xmin) + np.log10(xmax)) / 2, (np.log10(ymin) + np.log10(ymax)) / 2

    label_pos = []
    i = 0
    for line in contours.collections:
        for path in line.get_paths():
            logvert = np.log10(path.vertices)

            # Align with same x-value
            if levels[i] > 1:
                log_rho = -4.5
            else:
                log_rho = 15

            # logmid = log_rho, np.log10(levels[i]) - 2 * log_rho / 3
            i += 1

            # find closest point
            logdist = np.linalg.norm(logvert - logmid, ord=2, axis=1)
            min_ind = np.argmin(logdist)
            label_pos.append(10 ** logvert[min_ind, :])

    # Draw contour labels
    axes.clabel(
        contours,
        inline=True,
        inline_spacing=3,
        rightside_up=True,
        colors='red',
        fontsize=5,
        fmt=fmt,
        manual=label_pos,
    )


def get_axis_tables():
    """
    Grabs the cooling table information
    """
    g = h5.File(cooling_table, "r")

    density_bins = g["/TableBins/DensityBins"][:]
    U_bins = g["/TableBins/InternalEnergyBins"][:]
    Z_bins = g["/TableBins/MetallicityBins"][:]
    z_bins = g["/TableBins/RedshiftBins"][:]
    T_bins = g["/TableBins/TemperatureBins"][:]

    return density_bins, U_bins, Z_bins, z_bins, T_bins


def get_cooling_rates():
    """
    Grabs the cooling table information
    """
    g = h5.File(cooling_table, "r")

    return g["/Tdep/Cooling"]


def get_heating_rates():
    """
    Grabs the cooling table information
    """
    g = h5.File(cooling_table, "r")

    return g["/Tdep/Heating"]


def calculate_mean_cooling_times(data):
    data_cooling = get_cooling_rates()
    data_heating = get_heating_rates()

    cooling_rates = np.log10(np.power(10., data_cooling[0, :, :, :, -2]) + np.power(10., data_cooling[0, :, :, :, -1]))
    heating_rates = np.log10(np.power(10., data_heating[0, :, :, :, -2]) + np.power(10., data_heating[0, :, :, :, -1]))

    net_rates = np.log10(np.abs(np.power(10., heating_rates) - np.power(10., cooling_rates)))

    axis = get_axis_tables()
    nH_grid = axis[0]
    T_grid = axis[4]
    Z_grid = axis[2]

    f_net_rates = sci.RegularGridInterpolator(
        (T_grid, Z_grid, nH_grid),
        net_rates,
        method="linear",
        bounds_error=False,
        fill_value=-30
    )

    hydrogen_fraction = data.gas.element_mass_fractions.hydrogen
    gas_nH = (data.gas.densities / mh * hydrogen_fraction).to(cm ** -3)
    log_gas_nH = np.log10(gas_nH)
    temperature = data.gas.temperatures
    log_gas_T = np.log10(temperature)

    with np.errstate(divide='ignore'):
        log_gas_Z = np.log10(data.gas.metal_mass_fractions.value / solar_metallicity)

    # Values that go over the interpolation range are clipped to 0.5 Zsun
    if (log_gas_Z > 0.5).any():
        if args.debug:
            warn((
                f"[#] Found {(log_gas_Z > 0.5).sum()} particles above the upper "
                "metallicity bound in the interpolation tables. Values of "
                "log10(Z/Zsun) > 0.5 are capped to 0.5 for the calculation of "
                "net cooling times."
            ), RuntimeWarning)
        log_gas_Z[log_gas_Z > 0.5] = 0.5

    if (data.gas.metal_mass_fractions.value == 0).any():
        if args.debug:
            warn((
                f"[#] Found {(data.gas.metal_mass_fractions.value == 0).sum()} "
                "particles below the lower "
                "metallicity bound in the interpolation tables. Values of "
                "log10(Z/Zsun) < -50 are floored to -50 for the calculation of "
                "net cooling times."
            ), RuntimeWarning)
        log_gas_Z[data.gas.metal_mass_fractions.value == 0] = -50

    # construct the matrix that we input in the interpolator
    values_to_int = np.zeros((len(log_gas_T), 3))
    values_to_int[:, 0] = log_gas_T
    values_to_int[:, 1] = log_gas_Z
    values_to_int[:, 2] = log_gas_nH

    net_rates_found = f_net_rates(values_to_int)

    cooling_times = np.log10(3. / 2. * 1.38e-16) + log_gas_T - log_gas_nH - net_rates_found - np.log10(3.154e13)

    return cooling_times


def draw_cooling_contours(axes, density_bins, temperature_bins,
                          levels=[1.e4], color='green', prefix='', use_labels=True):
    data_cooling = get_cooling_rates()
    data_heating = get_heating_rates()

    cooling_rates = np.log10(np.power(10., data_cooling[0, :, :, :, -2]) + np.power(10., data_cooling[0, :, :, :, -1]))
    heating_rates = np.log10(np.power(10., data_heating[0, :, :, :, -2]) + np.power(10., data_heating[0, :, :, :, -1]))

    net_rates = np.log10(np.abs(np.power(10., heating_rates) - np.power(10., cooling_rates)))

    axis = get_axis_tables()
    nH_grid = axis[0]
    T_grid = axis[4]
    Z_grid = axis[2]

    f_net_rates = sci.RegularGridInterpolator(
        (T_grid, Z_grid, nH_grid),
        net_rates,
        method="linear",
        bounds_error=False,
        fill_value=-30
    )

    _density_interps, _temperature_interps = np.meshgrid(density_bins, temperature_bins)
    density_interps = _density_interps.flatten()
    temperature_interps = _temperature_interps.flatten()

    log_gas_nH = np.log10(density_interps)
    log_gas_T = np.log10(temperature_interps)
    log_gas_Z = np.ones_like(temperature_interps) * np.log10(1 / 3)

    # construct the matrix that we input in the interpolator
    values_to_int = np.zeros((len(log_gas_T), 3))
    values_to_int[:, 0] = log_gas_T
    values_to_int[:, 1] = log_gas_Z
    values_to_int[:, 2] = log_gas_nH

    net_rates_found = f_net_rates(values_to_int)

    cooling_time = np.log10(3. / 2. * 1.38e-16) + log_gas_T - log_gas_nH - net_rates_found - np.log10(3.154e13)
    cooling_time = cooling_time.reshape(_density_interps.shape)

    # Define entropy levels to plot
    _levels = np.log10(np.array(levels))

    contours = axes.contour(
        _density_interps,
        _temperature_interps,
        cooling_time,
        _levels,
        colors=color,
        linewidths=0.3,
        alpha=0.5
    )

    if use_labels:
        fmt = {value: f'{prefix}${latex_float(10 ** value)}$ Myr' for value in _levels}

        # work with logarithms for loglog scale
        # middle of the figure:
        xmin, xmax, ymin, ymax = axes.axis()
        logmid = (np.log10(xmin) + np.log10(xmax)) / 2, (np.log10(ymin) + np.log10(ymax)) / 2

        label_pos = []
        i = 0
        for line in contours.collections:
            for path in line.get_paths():
                logvert = np.log10(path.vertices)
                i += 1

                # find closest point
                logdist = np.linalg.norm(logvert - logmid, ord=2, axis=1)
                min_ind = np.argmin(logdist)
                label_pos.append(10 ** logvert[min_ind, :])

        # Draw contour labels
        axes.clabel(
            contours,
            inline=True,
            inline_spacing=3,
            rightside_up=True,
            colors=color,
            fontsize=5,
            fmt=fmt,
            manual=label_pos,
        )


def int_ticks(cbar):
    cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
    cbar.ax.yaxis.set_minor_formatter(ScalarFormatter())

    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    cbar.ax.yaxis.set_minor_locator(MaxNLocator(integer=True))


def draw_2d_hist(axes, x, y, z, cmap, label):
    if (z > 0).any():
        vmax = np.max(z) + 1
        mappable = axes.pcolormesh(
            x, y, z.T,
            norm=LogNorm(vmin=1, vmax=vmax),
            cmap=cmap,
            alpha=1
        )
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="3%", pad=0.)
        cbar = plt.colorbar(mappable, ax=axes, cax=cax)
        # int_ticks(cbar)
    else:
        axes.text(
            0.5, 0.5, 'Nothing here',
            transform=axes.transAxes,
            fontsize=20, color='gray', alpha=0.5,
            ha='center', va='center', rotation='30'
        )

    txt = AnchoredText(
        label, loc="upper right",
        frameon=False, pad=0.4, borderpad=0,
        prop={"fontsize": 8}
    )
    axes.add_artist(txt)

    # Star formation threshold
    axes.axvline(0.1, color='k', linestyle=':', lw=1, zorder=0)
    axes.set_xlabel(r"Density [$n_H$ cm$^{-3}$]")
    axes.set_ylabel(r"Temperature [K]")


class CoolingTimes(HaloProperty):

    def __init__(self):
        super().__init__()

    def process_single_halo(
            self,
            zoom_obj: Zoom = None,
            path_to_snap: str = None,
            path_to_catalogue: str = None,
            agn_time: str = None,
            z_agn_start: float = 18,
            z_agn_end: float = 0.,
    ):
        aperture_fraction = args.aperture_percent / 100

        sw_data, vr_data = self.get_handles_from_zoom(
            zoom_obj,
            path_to_snap,
            path_to_catalogue,
            mask_radius_r500=5
        )

        try:
            m500 = vr_data.spherical_overdensities.mass_500_rhocrit[0].to('Msun')
            r500 = vr_data.spherical_overdensities.r_500_rhocrit[0].to('Mpc')

        except AttributeError as err:
            print(err)
            print(f'[{self.__class__.__name__}] Launching spherical overdensity calculation...')
            spherical_overdensity = SODelta500(
                path_to_snap=path_to_snap,
                path_to_catalogue=path_to_catalogue,
            )
            m500 = spherical_overdensity.get_m500()
            r500 = spherical_overdensity.get_r500()

        if args.mass_estimator == 'hse':
            true_hse = HydrostaticEstimator(
                path_to_catalogue=path_to_catalogue,
                path_to_snap=path_to_snap,
                profile_type='true',
                diagnostics_on=False
            )
            r500 = true_hse.R500hse
            m500 = true_hse.M500hse

        aperture_fraction = aperture_fraction * r500

        # Convert datasets to physical quantities
        # R500c is already in physical units
        sw_data.gas.radial_distances.convert_to_physical()
        sw_data.gas.coordinates.convert_to_physical()
        sw_data.gas.masses.convert_to_physical()
        sw_data.gas.densities.convert_to_physical()

        activate_cooling_times = True
        try:
            cooling_times = calculate_mean_cooling_times(sw_data)
        except AttributeError as err:
            print(err)
            if args.debug:
                print(f'[{self.__class__.__name__}] Setting activate_cooling_times = False')
            activate_cooling_times = False

        if args.debug:
            print(f"[{self.__class__.__name__}] m500 = ", m500)
            print(f"[{self.__class__.__name__}] r500 = ", r500)
            print(f"[{self.__class__.__name__}] aperture_fraction = ", aperture_fraction)
            print(f"[{self.__class__.__name__}] Number of particles being imported", len(sw_data.gas.densities))

        gamma = 5 / 3

        try:
            a_heat = sw_data.gas.last_agnfeedback_scale_factors
        except AttributeError as err:
            print(err)
            print('Setting `last_agnfeedback_scale_factors` with 0.1.')
            a_heat = np.ones_like(sw_data.gas.masses) * 0.1

        try:
            fof_ids = sw_data.gas.fofgroup_ids
        except AttributeError as err:
            print(err)
            print(f"[{self.__class__.__name__}] Select particles only by radial distance.")
            fof_ids = np.ones_like(sw_data.gas.densities)

        try:
            temperature = sw_data.gas.temperatures
        except AttributeError as err:
            print(err)
            print(f"[{self.__class__.__name__}] Computing gas temperature from internal energies.")
            A = sw_data.gas.entropies * sw_data.units.mass
            temperature = mean_molecular_weight * (gamma - 1) * (A * sw_data.gas.densities ** (5 / 3 - 1)) / (
                    gamma - 1) * mh / kb

        try:
            hydrogen_fractions = sw_data.gas.element_mass_fractions.hydrogen
        except AttributeError as err:
            print(err)
            print(f"[{self.__class__.__name__}] Setting H fractions to primordial values.")
            hydrogen_fractions = np.ones_like(sw_data.gas.densities) * primordial_hydrogen_mass_fraction


        if agn_time is None:
            index = np.where(
                (sw_data.gas.radial_distances < aperture_fraction) &
                (fof_ids == 1) &
                (temperature > 1e5)
            )[0]

            number_density = (sw_data.gas.densities / mh).to('cm**-3').value[index] * hydrogen_fractions[index]
            temperature = temperature.to('K').value[index]

        elif agn_time == 'before':

            index = np.where(
                (sw_data.gas.radial_distances < aperture_fraction) &
                (fof_ids == 1) &
                (a_heat > (1 / (z_agn_start + 1))) &
                (a_heat < (1 / (z_agn_end + 1))) &
                (sw_data.gas.densities_before_last_agnevent > 0)
            )[0]

            density = sw_data.gas.densities_before_last_agnevent[index]
            number_density = (density / mh).to('cm**-3').value * hydrogen_fractions[index]
            A = sw_data.gas.entropies_before_last_agnevent[index] * sw_data.units.mass
            temperature = mean_molecular_weight * (gamma - 1) * (A * density ** (5 / 3 - 1)) / (
                    gamma - 1) * mh / kb
            temperature = temperature.to('K').value

        elif agn_time == 'after':

            index = np.where(
                (sw_data.gas.radial_distances < aperture_fraction) &
                (fof_ids == 1) &
                (a_heat > (1 / (z_agn_start + 1))) &
                (a_heat < (1 / (z_agn_end + 1))) &
                (sw_data.gas.densities_at_last_agnevent > 0)
            )[0]

            density = sw_data.gas.densities_at_last_agnevent[index]
            number_density = (density / mh).to('cm**-3').value * hydrogen_fractions[index]
            A = sw_data.gas.entropies_at_last_agnevent[index] * sw_data.units.mass
            temperature = mean_molecular_weight * (gamma - 1) * (A * density ** (5 / 3 - 1)) / (
                    gamma - 1) * mh / kb
            temperature = temperature.to('K').value

        agn_flag = sw_data.gas.heated_by_agnfeedback[index]
        snii_flag = sw_data.gas.heated_by_sniifeedback[index]
        agn_flag = agn_flag > 0
        snii_flag = snii_flag > 0

        # Calculate the critical density for the cross-hair marker
        rho_crit = unyt_quantity(
            sw_data.metadata.cosmology.critical_density(sw_data.metadata.z).value, 'g/cm**3'
        ).to('Msun/Mpc**3')
        nH_500 = (primordial_hydrogen_mass_fraction * Cosmology().fb * rho_crit * 500 / mh).to('cm**-3')

        # Entropy
        electron_number_density = (sw_data.gas.densities[index] / mh).to('cm**-3') / mean_molecular_weight
        entropy = kb * sw_data.gas.temperatures[index] / electron_number_density ** (2 / 3)
        entropy = entropy.to('keV*cm**2')

        x = number_density
        y = temperature

        if activate_cooling_times:
            w = cooling_times[index]

        if args.debug:
            print("Number of particles being plotted", len(x))

        # Set the limits of the figure.
        assert (x > 0).all(), f"Found negative value(s) in x: {x[x <= 0]}"
        assert (y > 0).all(), f"Found negative value(s) in y: {y[y <= 0]}"

        # density_bounds = [1e-6, 1e4]  # in nh/cm^3
        # temperature_bounds = [1e3, 1e10]  # in K
        density_bounds = [1e-5, 1]  # in nh/cm^3
        temperature_bounds = [1e6, 1e9]  # in K
        pdf_ybounds = [1, 10 ** 6]
        bins = 256

        # Make the norm object to define the image stretch
        density_bins = np.logspace(
            np.log10(density_bounds[0]), np.log10(density_bounds[1]), bins
        )
        temperature_bins = np.logspace(
            np.log10(temperature_bounds[0]), np.log10(temperature_bounds[1]), bins
        )

        T500 = (G * mean_molecular_weight * m500 * mp / r500 / 2 / kb).to('K').value
        K500 = (T500 * K * kb / (3 * m500 * Cosmology().fb / (4 * np.pi * r500 ** 3 * mp)) ** (
                2 / 3)).to('keV*cm**2')

        # Make the norm object to define the image stretch
        contour_density_bins = np.logspace(
            np.log10(density_bounds[0]) - 0.5, np.log10(density_bounds[1]) + 0.5, bins * 4
        )
        contour_temperature_bins = np.logspace(
            np.log10(temperature_bounds[0]) - 0.5, np.log10(temperature_bounds[1]) + 0.5, bins * 4
        )

        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.7)
        axes = gs.subplots()

        for ax in [
            axes[0, 0],
            axes[0, 1],
            axes[0, 2],
            axes[1, 0],
            axes[1, 1]
        ]:
            ax.loglog()
            # Draw cross-hair marker
            ax.hlines(y=T500, xmin=nH_500 / 3, xmax=nH_500 * 3, colors='k', linestyles='-', lw=0.5)
            ax.vlines(x=nH_500, ymin=T500 / 5, ymax=T500 * 5, colors='k', linestyles='-', lw=0.5)

            # Draw contours
            draw_k500(ax, contour_density_bins, contour_temperature_bins, K500)
            draw_adiabats(ax, contour_density_bins, contour_temperature_bins)
            draw_cooling_contours(ax, contour_density_bins, contour_temperature_bins,
                                  levels=[1, 1e2, 1e3, 1e4, 1e5],
                                  color='green')
            draw_cooling_contours(ax, contour_density_bins, contour_temperature_bins,
                                  levels=[Cosmology().age(sw_data.metadata.z).to('Myr').value],
                                  prefix='$t_H(z)=$',
                                  color='red',
                                  use_labels=False)

        # PLOT ALL PARTICLES ===============================================
        H, density_edges, temperature_edges = np.histogram2d(
            x, y, bins=[density_bins, temperature_bins]
        )
        draw_2d_hist(axes[0, 0], density_edges, temperature_edges, H, 'Greys_r', "All particles")

        # PLOT SN HEATED PARTICLES ===============================================
        H, density_edges, temperature_edges = np.histogram2d(
            x[(snii_flag & ~agn_flag)],
            y[(snii_flag & ~agn_flag)],
            bins=[density_bins, temperature_bins]
        )
        draw_2d_hist(axes[0, 1], density_edges, temperature_edges, H, 'Greens_r', "SNe heated only")
        axes[0, 1].axhline(10 ** 7.5, color='k', linestyle='--', lw=1, zorder=0)

        # PLOT NOT HEATED PARTICLES ===============================================
        H, density_edges, temperature_edges = np.histogram2d(
            x[(~snii_flag & ~agn_flag)],
            y[(~snii_flag & ~agn_flag)],
            bins=[density_bins, temperature_bins]
        )
        draw_2d_hist(axes[0, 2], density_edges, temperature_edges, H, 'Greens_r', "Not heated")

        # PLOT AGN HEATED PARTICLES ===============================================
        H, density_edges, temperature_edges = np.histogram2d(
            x[(agn_flag & ~snii_flag)],
            y[(agn_flag & ~snii_flag)],
            bins=[density_bins, temperature_bins]
        )
        draw_2d_hist(axes[1, 1], density_edges, temperature_edges, H, 'Reds_r', "AGN heated only")
        axes[1, 1].axhline(10 ** 8.5, color='k', linestyle='--', lw=1, zorder=0)

        # PLOT AGN+SN HEATED PARTICLES ===============================================
        H, density_edges, temperature_edges = np.histogram2d(
            x[(agn_flag & snii_flag)],
            y[(agn_flag & snii_flag)],
            bins=[density_bins, temperature_bins]
        )
        draw_2d_hist(axes[1, 0], density_edges, temperature_edges, H, 'Purples_r', "AGN and SNe heated")
        axes[1, 0].axhline(10 ** 8.5, color='k', linestyle='--', lw=1, zorder=0)
        axes[1, 0].axhline(10 ** 7.5, color='k', linestyle='--', lw=1, zorder=0)

        bins = np.linspace(0., 5.5, 51)

        axes[2, 0].clear()
        axes[2, 0].set_xscale('linear')
        axes[2, 0].set_yscale('log')
        if activate_cooling_times:
            axes[2, 0].hist(w, bins=bins, histtype='step', label='All')
            axes[2, 0].hist(w[(agn_flag & snii_flag)], bins=bins, histtype='step', label='AGN & SN')
            axes[2, 0].hist(w[(agn_flag & ~snii_flag)], bins=bins, histtype='step', label='AGN')
            axes[2, 0].hist(w[(~agn_flag & snii_flag)], bins=bins, histtype='step', label='SN')
            axes[2, 0].hist(w[(~agn_flag & ~snii_flag)], bins=bins, histtype='step', label='Not heated')
        axes[2, 0].axvline(np.log10(Cosmology().age(sw_data.metadata.z).to('Myr').value),
                           color='k', linestyle='--', lw=0.5, zorder=0)
        axes[2, 0].set_xlabel(f"$\log_{{10}}$(Cooling time [Myr])")
        axes[2, 0].set_ylabel('Number of particles')
        axes[2, 0].set_ylim(pdf_ybounds)
        axes[2, 0].legend(loc="upper left")

        if activate_cooling_times:
            hydrogen_fraction = sw_data.gas.element_mass_fractions.hydrogen[index]
        bins = np.linspace(0, 1, 51)

        axes[2, 1].clear()
        axes[2, 1].set_xscale('linear')
        axes[2, 1].set_yscale('log')
        if activate_cooling_times:
            axes[2, 1].hist(hydrogen_fraction, bins=bins, histtype='step', label='All')
            axes[2, 1].hist(hydrogen_fraction[(agn_flag & snii_flag)], bins=bins, histtype='step', label='AGN & SN')
            axes[2, 1].hist(hydrogen_fraction[(agn_flag & ~snii_flag)], bins=bins, histtype='step', label='AGN')
            axes[2, 1].hist(hydrogen_fraction[(~agn_flag & snii_flag)], bins=bins, histtype='step', label='SN')
            axes[2, 1].hist(hydrogen_fraction[(~agn_flag & ~snii_flag)], bins=bins, histtype='step', label='Not heated')
        axes[2, 1].set_xlabel("Hydrogen fraction")
        axes[2, 1].set_ylabel('Number of particles')
        axes[2, 1].set_ylim(pdf_ybounds)

        if activate_cooling_times:
            log_gas_Z = np.log10(sw_data.gas.metal_mass_fractions.value[index] / 0.0133714)
        bins = np.linspace(-4, 1, 51)

        axes[2, 2].clear()
        axes[2, 2].set_xscale('linear')
        axes[2, 2].set_yscale('log')
        if activate_cooling_times:
            axes[2, 2].hist(log_gas_Z, bins=bins, histtype='step', label='All')
            axes[2, 2].hist(log_gas_Z[(agn_flag & snii_flag)], bins=bins, histtype='step', label='AGN & SN')
            axes[2, 2].hist(log_gas_Z[(agn_flag & ~snii_flag)], bins=bins, histtype='step', label='AGN')
            axes[2, 2].hist(log_gas_Z[(~agn_flag & snii_flag)], bins=bins, histtype='step', label='SN')
            axes[2, 2].hist(log_gas_Z[(~agn_flag & ~snii_flag)], bins=bins, histtype='step', label='Not heated')
        axes[2, 2].axvline(0.5, color='k', linestyle='--', lw=0.5, zorder=0)
        axes[2, 2].set_xlabel(f"$\log_{{10}}$(Metallicity [Z$_\odot$])")
        axes[2, 2].set_ylabel('Number of particles')
        axes[2, 2].set_ylim(pdf_ybounds)

        bins = np.logspace(np.log10(density_edges.min()), np.log10(density_edges.max()), 51)

        axes[0, 3].clear()
        axes[0, 3].set_xscale('log')
        axes[0, 3].set_yscale('log')
        axes[0, 3].hist(x, bins=bins, histtype='step', label='All')
        axes[0, 3].hist(x[(agn_flag & snii_flag)], bins=bins, histtype='step', label='AGN & SN')
        axes[0, 3].hist(x[(agn_flag & ~snii_flag)], bins=bins, histtype='step', label='AGN')
        axes[0, 3].hist(x[(~agn_flag & snii_flag)], bins=bins, histtype='step', label='SN')
        axes[0, 3].hist(x[(~agn_flag & ~snii_flag)], bins=bins, histtype='step', label='Not heated')
        axes[0, 3].set_xlabel(f"Density [$n_H$ cm$^{{-3}}$]")
        axes[0, 3].set_ylabel('Number of particles')
        axes[0, 3].set_ylim(pdf_ybounds)
        axes[0, 3].legend()

        bins = np.logspace(np.log10(temperature_edges.min()), np.log10(temperature_edges.max()), 51)

        axes[1, 3].clear()
        axes[1, 3].set_xscale('log')
        axes[1, 3].set_yscale('log')
        axes[1, 3].hist(y, bins=bins, histtype='step', label='All')
        axes[1, 3].hist(y[(agn_flag & snii_flag)], bins=bins, histtype='step', label='AGN & SN')
        axes[1, 3].hist(y[(agn_flag & ~snii_flag)], bins=bins, histtype='step', label='AGN')
        axes[1, 3].hist(y[(~agn_flag & snii_flag)], bins=bins, histtype='step', label='SN')
        axes[1, 3].hist(y[(~agn_flag & ~snii_flag)], bins=bins, histtype='step', label='Not heated')
        axes[1, 3].set_xlabel("Temperature [K]")
        axes[1, 3].set_ylabel('Number of particles')
        axes[1, 3].set_ylim(pdf_ybounds)

        bins = np.logspace(0, 4, 51)

        axes[2, 3].clear()
        axes[2, 3].set_xscale('log')
        axes[2, 3].set_yscale('log')
        axes[2, 3].hist(entropy, bins=bins, histtype='step', label='All')
        axes[2, 3].hist(entropy[(agn_flag & snii_flag)], bins=bins, histtype='step', label='AGN & SN')
        axes[2, 3].hist(entropy[(agn_flag & ~snii_flag)], bins=bins, histtype='step', label='AGN')
        axes[2, 3].hist(entropy[(~agn_flag & snii_flag)], bins=bins, histtype='step', label='SN')
        axes[2, 3].hist(entropy[(~agn_flag & ~snii_flag)], bins=bins, histtype='step', label='Not heated')
        axes[2, 3].set_xlabel("Entropy [keV cm$^2$]")
        axes[2, 3].set_ylabel('Number of particles')
        axes[2, 3].set_ylim(pdf_ybounds)

        # Entropy profile
        max_radius_r500 = 4
        index = np.where(
            (sw_data.gas.radial_distances < max_radius_r500 * r500) &
            (fof_ids == 1) &
            (sw_data.gas.temperatures > 1e5)
        )[0]
        radial_distance = sw_data.gas.radial_distances[index] / r500
        sw_data.gas.masses = sw_data.gas.masses[index]
        sw_data.gas.temperatures = sw_data.gas.temperatures[index]

        # Define radial bins and shell volumes
        lbins = np.logspace(-2, np.log10(max_radius_r500), 51) * radial_distance.units
        radial_bin_centres = 10.0 ** (0.5 * np.log10(lbins[1:] * lbins[:-1])) * radial_distance.units
        volume_shell = (4. * np.pi / 3.) * (r500 ** 3) * ((lbins[1:]) ** 3 - (lbins[:-1]) ** 3)

        mass_weights, _ = histogram_unyt(radial_distance, bins=lbins, weights=sw_data.gas.masses)
        mass_weights[mass_weights == 0] = np.nan  # Replace zeros with Nans
        density_profile = mass_weights / volume_shell
        number_density_profile = (density_profile.to('g/cm**3') / (mp * mean_molecular_weight)).to('cm**-3')

        mass_weighted_temperatures = (sw_data.gas.temperatures * kb).to('keV') * sw_data.gas.masses
        temperature_weights, _ = histogram_unyt(radial_distance, bins=lbins, weights=mass_weighted_temperatures)
        temperature_weights[temperature_weights == 0] = np.nan  # Replace zeros with Nans
        temperature_profile = temperature_weights / mass_weights  # kBT in units of [keV]

        entropy_profile = temperature_profile / number_density_profile ** (2 / 3)

        rho_crit = unyt_quantity(
            sw_data.metadata.cosmology.critical_density(sw_data.metadata.z).value, 'g/cm**3'
        ).to('Msun/Mpc**3')
        density_profile /= rho_crit

        kBT500 = (
                G * mean_molecular_weight * m500 * mp / r500 / 2
        ).to('keV')
        K500 = (
                kBT500 / (3 * m500 * Cosmology().fb / (4 * np.pi * r500 ** 3 * mp)) ** (2 / 3)
        ).to('keV*cm**2')

        axes[1, 2].plot(
            radial_bin_centres,
            entropy_profile,
            linestyle='-',
            color='r',
            linewidth=1,
            alpha=1,
        )
        axes[1, 2].set_xscale('log')
        axes[1, 2].set_yscale('log')
        axes[1, 2].axhline(y=K500, color='k', linestyle=':', linewidth=0.5)
        axes[1, 2].axvline(0.15, color='k', linestyle='--', lw=0.5, zorder=0)
        axes[1, 2].set_ylabel(r'Entropy [keV cm$^2$]')
        axes[1, 2].set_xlabel(r'$r/r_{500}$')
        axes[1, 2].set_ylim([1, 1e4])
        axes[1, 2].set_xlim([0.01, max_radius_r500])
        axes[1, 2].text(
            axes[1, 2].get_xlim()[0], K500, r'$K_{500}$',
            horizontalalignment='left',
            verticalalignment='bottom',
            color='k',
            bbox=dict(
                boxstyle='square,pad=10',
                fc='none',
                ec='none'
            )
        )
        sun_observations = Sun2009()
        sun_observations.filter_by('M_500', 8e13, 3e14)
        sun_observations.overlay_entropy_profiles(
            axes=axes[1, 2],
            k_units='keVcm^2',
            markersize=1,
            linewidth=0.5
        )
        rexcess = Pratt2010()
        bin_median, bin_perc16, bin_perc84 = rexcess.combine_entropy_profiles(
            m500_limits=(
                1e14 * Solar_Mass,
                5e14 * Solar_Mass
            ),
            k500_rescale=False
        )
        axes[1, 2].fill_between(
            rexcess.radial_bins,
            bin_perc16,
            bin_perc84,
            color='aqua',
            alpha=0.85,
            linewidth=0
        )
        axes[1, 2].plot(rexcess.radial_bins, bin_median, c='k')

        z_agn_recent_text = (
            f"Selecting gas heated between {z_agn_start:.1f} > z > {z_agn_end:.1f} (relevant to AGN plot only)\n"
            f"({1 / (z_agn_start + 1):.2f} < a < {1 / (z_agn_end + 1):.2f})\n"
        )
        if agn_time is not None:
            z_agn_recent_text = (
                f"Selecting gas {agn_time:s} heated between {z_agn_start:.1f} > z > {z_agn_end:.1f}\n"
                f"({1 / (z_agn_start + 1):.2f} < a < {1 / (z_agn_end + 1):.2f})\n"
            )

        fig.suptitle(
            (
                f"{os.path.basename(path_to_snap)}\n"
                f"Aperture = {args.aperture_percent / 100:.2f} $R_{{500}}$\t\t"
                f"$z = {sw_data.metadata.z:.2f}$\t\t"
                f"Age = {Cosmology().age(sw_data.metadata.z).value:.2f} Gyr\t\t"
                f"\t$M_{{500}}={latex_float(m500.value)}\\ {m500.units.latex_repr}$\n"
                f"{z_agn_recent_text:s}"
                f"Central FoF group only\t\tEstimator: {args.mass_estimator}"
            ),
            fontsize=7
        )

        if not args.quiet:
            plt.show()

        fig.savefig(
            os.path.join(
                default_output_directory,
                f"cooling_times_{os.path.basename(path_to_snap)[:-5].replace('.', 'p')}.png"
            ),
            dpi=300
        )

        plt.close()
