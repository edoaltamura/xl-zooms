import numpy as np
import os
import h5py as h5
import scipy.interpolate as sci
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable
from unyt import *
from swiftsimio.visualisation.projection import project_pixel_grid
import swiftsimio

from literature import Cosmology
from register import Zoom, args, cooling_table, default_output_directory
from .halo_property import HaloProperty

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


def int_ticks(cbar):
    # Major ticks
    ticks = cbar.ax.get_yticks(minor=False)
    labels = cbar.ax.get_yticklabels(minor=False)

    num_majors = len(labels)

    for t, l in zip(ticks, labels):

        if float(t) < 10:
            l.set_text(f'{int(float(t))}')
        else:
            l.set_text(f'$10^{{{int(np.log10(float(t)))}}}$')

    cbar.ax.set_yticks(ticks, minor=False)
    cbar.ax.set_yticklabels(labels, minor=False)

    # Minor ticks
    ticks = cbar.ax.get_yticks(minor=True)
    labels = cbar.ax.get_yticklabels(minor=True)

    for t, l in zip(ticks, labels):

        if float(t) < 100 and num_majors < 2:
            if float(t) < 1.01 or float(t) > 1.99:
                l.set_text(f'{int(float(t))}')

    cbar.ax.set_yticks(ticks, minor=True)
    cbar.ax.set_yticklabels(labels, minor=True)

    return cbar


def draw_adiabats(axes, density_bins, temperature_bins):
    density_interps, temperature_interps = np.meshgrid(density_bins, temperature_bins)
    entropy_interps = temperature_interps * K * boltzmann_constant / (density_interps / cm ** 3) ** (2 / 3)
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
    entropy_interps = temperature_interps * K * boltzmann_constant / (density_interps / cm ** 3) ** (2 / 3)
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
        print((
            f"[#] Found {(log_gas_Z > 0.5).sum()} particles above the upper "
            "metallicity bound in the interpolation tables. Values of "
            "log10(Z/Zsun) > 0.5 are capped to 0.5 for the calculation of "
            "net cooling times."
        ))
        log_gas_Z[log_gas_Z > 0.5] = 0.5

    if (data.gas.metal_mass_fractions.value == 0).any():
        print((
            f"[#] Found {(data.gas.metal_mass_fractions.value == 0).sum()} "
            "particles below the lower "
            "metallicity bound in the interpolation tables. Values of "
            "log10(Z/Zsun) < -50 are floored to -50 for the calculation of "
            "net cooling times."
        ))
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
            **kwargs
    ):
        aperture_fraction = args.aperture_percent / 100
        sw_data, vr_data = self.get_handles_from_zoom(zoom_obj, path_to_snap, path_to_catalogue,
                                                      mask_radius_r500=aperture_fraction, **kwargs)

        m500 = vr_data.spherical_overdensities.mass_500_rhocrit[0].to('Msun')
        r500 = vr_data.spherical_overdensities.r_500_rhocrit[0].to('Mpc')

        aperture_fraction = aperture_fraction * r500

        # Convert datasets to physical quantities
        # R500c is already in physical units
        sw_data.gas.radial_distances.convert_to_physical()
        sw_data.gas.coordinates.convert_to_physical()
        sw_data.gas.masses.convert_to_physical()
        sw_data.gas.densities.convert_to_physical()

        cooling_times = calculate_mean_cooling_times(sw_data)

        gamma = 5 / 3

        try:
            a_heat = sw_data.gas.last_agnfeedback_scale_factors
        except AttributeError as e:
            print(e)
            print('Setting `last_agnfeedback_scale_factors` with 0.1.')
            a_heat = np.ones_like(cooling_times) / 10

        if agn_time is None:

            if z_agn_start < 7.2 or z_agn_end > 0:

                index = np.where(
                    (sw_data.gas.radial_distances < aperture_fraction) &
                    (sw_data.gas.fofgroup_ids == 1) &
                    (a_heat > (1 / (z_agn_start + 1))) &
                    (a_heat < (1 / (z_agn_end + 1))) &
                    (sw_data.gas.temperatures > 1e5) &
                    (cooling_times > 0)
                )[0]
            else:
                index = np.where(
                    (sw_data.gas.radial_distances < aperture_fraction) &
                    (sw_data.gas.fofgroup_ids == 1) &
                    (sw_data.gas.temperatures > 1e5) &
                    (cooling_times > 0)
                )[0]

            number_density = (sw_data.gas.densities / mh).to('cm**-3').value[index] * \
                             sw_data.gas.element_mass_fractions.hydrogen[index]
            temperature = sw_data.gas.temperatures.to('K').value[index]

        elif agn_time == 'before':

            index = np.where(
                (sw_data.gas.radial_distances < aperture_fraction) &
                (sw_data.gas.fofgroup_ids == 1) &
                (a_heat > (1 / (z_agn_start + 1))) &
                (a_heat < (1 / (z_agn_end + 1))) &
                (sw_data.gas.densities_before_last_agnevent > 0)
            )[0]

            density = sw_data.gas.densities_before_last_agnevent[index]
            number_density = (density / mh).to('cm**-3').value * sw_data.gas.element_mass_fractions.hydrogen[index]
            A = sw_data.gas.entropies_before_last_agnevent[index] * sw_data.units.mass
            temperature = mean_molecular_weight * (gamma - 1) * (A * density ** (5 / 3 - 1)) / (
                    gamma - 1) * mh / boltzmann_constant
            temperature = temperature.to('K').value

        elif agn_time == 'after':

            index = np.where(
                (sw_data.gas.radial_distances < aperture_fraction) &
                (sw_data.gas.fofgroup_ids == 1) &
                (a_heat > (1 / (z_agn_start + 1))) &
                (a_heat < (1 / (z_agn_end + 1))) &
                (sw_data.gas.densities_at_last_agnevent > 0)
            )[0]

            density = sw_data.gas.densities_at_last_agnevent[index]
            number_density = (density / mh).to('cm**-3').value * sw_data.gas.element_mass_fractions.hydrogen[index]
            A = sw_data.gas.entropies_at_last_agnevent[index] * sw_data.units.mass
            temperature = mean_molecular_weight * (gamma - 1) * (A * density ** (5 / 3 - 1)) / (
                    gamma - 1) * mh / boltzmann_constant
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

        x = number_density
        y = temperature
        w = cooling_times[index]

        print("Number of particles being plotted", len(x))

        # Set the limits of the figure.
        assert (x > 0).all(), f"Found negative value(s) in x: {x[x <= 0]}"
        assert (y > 0).all(), f"Found negative value(s) in y: {y[y <= 0]}"

        # density_bounds = [1e-6, 1e4]  # in nh/cm^3
        # temperature_bounds = [1e3, 1e10]  # in K
        density_bounds = [1e-4, 1]  # in nh/cm^3
        temperature_bounds = [1e6, 1e9]  # in K
        bins = 256

        # Make the norm object to define the image stretch
        density_bins = np.logspace(
            np.log10(density_bounds[0]), np.log10(density_bounds[1]), bins
        )
        temperature_bins = np.logspace(
            np.log10(temperature_bounds[0]), np.log10(temperature_bounds[1]), bins
        )

        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.4)
        axes = gs.subplots()

        for ax in axes.flat:
            ax.loglog()

            # Draw cross-hair marker
            T500 = (G * mean_molecular_weight * m500 * mp / r500 / 2 / boltzmann_constant).to('K').value
            ax.hlines(y=T500, xmin=nH_500 / 3, xmax=nH_500 * 3, colors='k', linestyles='-', lw=1)
            ax.vlines(x=nH_500, ymin=T500 / 5, ymax=T500 * 5, colors='k', linestyles='-', lw=1)
            K500 = (T500 * K * boltzmann_constant / (3 * m500 * Cosmology().fb / (4 * np.pi * r500 ** 3 * mp)) ** (
                    2 / 3)).to('keV*cm**2')

            # Make the norm object to define the image stretch
            contour_density_bins = np.logspace(
                np.log10(density_bounds[0]) - 0.5, np.log10(density_bounds[1]) + 0.5, bins * 4
            )
            contour_temperature_bins = np.logspace(
                np.log10(temperature_bounds[0]) - 0.5, np.log10(temperature_bounds[1]) + 0.5, bins * 4
            )

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

            # Star formation threshold
            ax.axvline(0.1, color='k', linestyle=':', lw=1, zorder=0)
            ax.set_xlabel(r"Density [$n_H$ cm$^{-3}$]")
            ax.set_ylabel(r"Temperature [K]")

        # PLOT ALL PARTICLES ===============================================
        H, density_edges, temperature_edges = np.histogram2d(
            x, y, bins=[density_bins, temperature_bins]
        )
        if (H > 0).any():
            vmax = np.max(H) + 1
            mappable = axes[0, 0].pcolormesh(
                density_edges, temperature_edges, H.T,
                norm=LogNorm(vmin=1, vmax=vmax), cmap='Greys_r'
            )
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(axes[0, 0])
            cax = divider.append_axes("right", size="3%", pad=0.)
            cbar = plt.colorbar(mappable, ax=axes[0, 0], cax=cax)
            int_ticks(cbar)
        else:
            axes[0, 1].text(0.5, 0.5, 'Nothing here', transform=axes[0, 1].transAxes,
                            fontsize=40, color='gray', alpha=0.5,
                            ha='center', va='center', rotation='30')

        txt = AnchoredText("All particles", loc="upper right", frameon=False, pad=0.4, borderpad=0,
                           prop={"fontsize": 8})
        axes[0, 0].add_artist(txt)

        # PLOT SN HEATED PARTICLES ===============================================
        H, density_edges, temperature_edges = np.histogram2d(
            x[(snii_flag & ~agn_flag)],
            y[(snii_flag & ~agn_flag)],
            bins=[density_bins, temperature_bins]
        )

        if (H > 0).any():
            vmax = np.max(H) + 1
            mappable = axes[0, 1].pcolormesh(
                density_edges, temperature_edges, H.T,
                norm=LogNorm(vmin=1, vmax=vmax), cmap='Greens_r', alpha=0.6
            )
            divider = make_axes_locatable(axes[0, 1])
            cax = divider.append_axes("right", size="3%", pad=0.)
            cbar = plt.colorbar(mappable, ax=axes[0, 1], cax=cax)
            int_ticks(cbar)
        else:
            axes[0, 1].text(0.5, 0.5, 'Nothing here', transform=axes[0, 1].transAxes,
                            fontsize=40, color='gray', alpha=0.5,
                            ha='center', va='center', rotation='30')

        # Heating temperatures
        axes[0, 1].axhline(10 ** 7.5, color='k', linestyle='--', lw=1, zorder=0)
        txt = AnchoredText("SNe heated only", loc="upper right", frameon=False, pad=0.4, borderpad=0,
                           prop={"fontsize": 8})
        axes[0, 1].add_artist(txt)

        # PLOT NOT HEATED PARTICLES ===============================================
        H, density_edges, temperature_edges = np.histogram2d(
            x[(~snii_flag & ~agn_flag)],
            y[(~snii_flag & ~agn_flag)],
            bins=[density_bins, temperature_bins]
        )

        if (H > 0).any():
            vmax = np.max(H) + 1
            mappable = axes[0, 2].pcolormesh(
                density_edges, temperature_edges, H.T,
                norm=LogNorm(vmin=1, vmax=vmax), cmap='Greens_r', alpha=0.6
            )
            divider = make_axes_locatable(axes[0, 2])
            cax = divider.append_axes("right", size="3%", pad=0.)
            cbar = plt.colorbar(mappable, ax=axes[0, 2], cax=cax)
            int_ticks(cbar)
        else:
            axes[0, 1].text(0.5, 0.5, 'Nothing here', transform=axes[0, 1].transAxes,
                            fontsize=40, color='gray', alpha=0.5,
                            ha='center', va='center', rotation='30')

        txt = AnchoredText("Not heated by SN or AGN", loc="upper right", frameon=False, pad=0.4, borderpad=0,
                           prop={"fontsize": 8})
        axes[0, 2].add_artist(txt)

        # PLOT AGN HEATED PARTICLES ===============================================
        H, density_edges, temperature_edges = np.histogram2d(
            x[(agn_flag & ~snii_flag)],
            y[(agn_flag & ~snii_flag)],
            bins=[density_bins, temperature_bins]
        )

        if (H > 0).any():
            vmax = np.max(H) + 1
            mappable = axes[1, 1].pcolormesh(
                density_edges, temperature_edges, H.T,
                norm=LogNorm(vmin=1, vmax=vmax), cmap='Reds_r', alpha=0.6
            )
            divider = make_axes_locatable(axes[1, 1])
            cax = divider.append_axes("right", size="3%", pad=0.)
            cbar = plt.colorbar(mappable, ax=axes[1, 1], cax=cax)
            int_ticks(cbar)
        else:
            axes[0, 1].text(0.5, 0.5, 'Nothing here', transform=axes[0, 1].transAxes,
                            fontsize=40, color='gray', alpha=0.5,
                            ha='center', va='center', rotation='30')

        txt = AnchoredText("AGN heated only", loc="upper right", frameon=False, pad=0.4, borderpad=0,
                           prop={"fontsize": 8})
        axes[1, 1].add_artist(txt)
        # Heating temperatures
        axes[1, 1].axhline(10 ** 8.5, color='k', linestyle='--', lw=1, zorder=0)

        # PLOT AGN+SN HEATED PARTICLES ===============================================
        H, density_edges, temperature_edges = np.histogram2d(
            x[(agn_flag & snii_flag)],
            y[(agn_flag & snii_flag)],
            bins=[density_bins, temperature_bins]
        )

        if (H > 0).any():
            vmax = np.max(H) + 1
            mappable = axes[1, 0].pcolormesh(
                density_edges, temperature_edges, H.T,
                norm=LogNorm(vmin=1, vmax=vmax), cmap='Purples_r', alpha=0.6
            )
            divider = make_axes_locatable(axes[1, 0])
            cax = divider.append_axes("right", size="3%", pad=0.)
            cbar = plt.colorbar(mappable, ax=axes[1, 0], cax=cax)
            int_ticks(cbar)
        else:
            axes[0, 1].text(0.5, 0.5, 'Nothing here', transform=axes[0, 1].transAxes,
                            fontsize=40, color='gray', alpha=0.5,
                            ha='center', va='center', rotation='30')

        txt = AnchoredText("AGN and SNe heated", loc="upper right", frameon=False, pad=0.4, borderpad=0,
                           prop={"fontsize": 8})
        axes[1, 0].add_artist(txt)
        # Heating temperatures
        axes[1, 0].axhline(10 ** 8.5, color='k', linestyle='--', lw=1, zorder=0)
        axes[1, 0].axhline(10 ** 7.5, color='k', linestyle='--', lw=1, zorder=0)

        axes[1, 2].remove()

        bins = np.linspace(w.min(), w.max(), 51)

        axes[2, 0].clear()
        axes[2, 0].set_xscale('linear')
        axes[2, 0].set_yscale('log')
        axes[2, 0].hist(w, bins=bins, histtype='step', label='All')
        axes[2, 0].hist(w[(agn_flag & snii_flag)], bins=bins, histtype='step', label='AGN & SN')
        axes[2, 0].hist(w[(agn_flag & ~snii_flag)], bins=bins, histtype='step', label='AGN')
        axes[2, 0].hist(w[(~agn_flag & snii_flag)], bins=bins, histtype='step', label='SN')
        axes[2, 0].hist(w[(~agn_flag & ~snii_flag)], bins=bins, histtype='step', label='Not heated')
        axes[2, 0].axvline(np.log10(Cosmology().age(sw_data.metadata.z).to('Myr').value),
                           color='k', linestyle='--', lw=0.5, zorder=0)
        axes[2, 0].set_xlabel(f"$\log_{{10}}$(Cooling time [Myr])")
        axes[2, 0].set_ylabel('Number of particles')
        axes[2, 0].legend(loc = "upper left")

        hydrogen_fraction = sw_data.gas.element_mass_fractions.hydrogen[index]
        bins = np.linspace(hydrogen_fraction.min(), hydrogen_fraction.max(), 51)

        axes[2, 1].clear()
        axes[2, 1].set_xscale('linear')
        axes[2, 1].set_yscale('log')
        axes[2, 1].hist(hydrogen_fraction, bins=bins, histtype='step', label='All')
        axes[2, 1].hist(hydrogen_fraction[(agn_flag & snii_flag)], bins=bins, histtype='step', label='AGN & SN')
        axes[2, 1].hist(hydrogen_fraction[(agn_flag & ~snii_flag)], bins=bins, histtype='step', label='AGN')
        axes[2, 1].hist(hydrogen_fraction[(~agn_flag & snii_flag)], bins=bins, histtype='step', label='SN')
        axes[2, 1].hist(hydrogen_fraction[(~agn_flag & ~snii_flag)], bins=bins, histtype='step', label='Not heated')
        axes[2, 1].set_xlabel("Hydrogen fraction")
        axes[2, 1].set_ylabel('Number of particles')

        log_gas_Z = np.log10(sw_data.gas.metal_mass_fractions.value[index] / 0.0133714)
        bins = np.linspace(log_gas_Z.min(), log_gas_Z.max(), 51)

        axes[2, 2].clear()
        axes[2, 2].set_xscale('linear')
        axes[2, 2].set_yscale('log')
        axes[2, 2].hist(log_gas_Z, bins=bins, histtype='step', label='All')
        axes[2, 2].hist(log_gas_Z[(agn_flag & snii_flag)], bins=bins, histtype='step', label='AGN & SN')
        axes[2, 2].hist(log_gas_Z[(agn_flag & ~snii_flag)], bins=bins, histtype='step', label='AGN')
        axes[2, 2].hist(log_gas_Z[(~agn_flag & snii_flag)], bins=bins, histtype='step', label='SN')
        axes[2, 2].hist(log_gas_Z[(~agn_flag & ~snii_flag)], bins=bins, histtype='step', label='Not heated')
        axes[2, 2].axvline(0.5, color='k', linestyle='--', lw=0.5, zorder=0)
        axes[2, 2].set_xlabel(f"$\log_{{10}}$(Metallicity [Z$_\odot$])")
        axes[2, 2].set_ylabel('Number of particles')

        bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 51)

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
        axes[0, 3].legend()

        bins = np.logspace(np.log10(y.min()), np.log10(y.max()), 51)

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

        # Density map
        _xCen = vr_data.positions.xcminpot[0].to('Mpc') / vr_data.a
        _yCen = vr_data.positions.ycminpot[0].to('Mpc') / vr_data.a
        _r500 = vr_data.spherical_overdensities.r_500_rhocrit[0].to('Mpc') / vr_data.a
        sw_handle = swiftsimio.load(path_to_snap)
        region = [
                _xCen - 3 * _r500,
                _xCen + 3 * _r500,
                _yCen - 3 * _r500,
                _yCen + 3 * _r500
            ]
        gas_mass = project_pixel_grid(
            # Note here that we pass in the dark matter dataset not the whole
            # data object, to specify what particle type we wish to visualise
            data=sw_handle.gas,
            boxsize=sw_handle.metadata.boxsize,
            resolution=1024,
            project='densities',
            parallel=True,
            region=region
        )

        axes[2, 3].axis("off")
        axes[2, 3].set_aspect("equal")
        axes[2, 3].imshow(gas_mass.T, norm=LogNorm(), cmap="twilight", origin="lower", extent=region)
        circle_r500 = plt.Circle((_xCen, _yCen), _r500, color="red", fill=False, linestyle='-')
        axes[2, 3].add_artist(circle_r500)

        z_agn_recent_text = (
            f"Selecting gas heated between {z_agn_start:.1f} < z < {z_agn_end:.1f} (relevant to AGN plot only)\n"
            f"({1 / (z_agn_start + 1):.2f} < a < {1 / (z_agn_end + 1):.2f})\n"
        )
        if agn_time is not None:
            z_agn_recent_text = (
                f"Selecting gas {agn_time:s} heated between {z_agn_start:.1f} < z < {z_agn_end:.1f}\n"
                f"({1 / (z_agn_start + 1):.2f} < a < {1 / (z_agn_end + 1):.2f})\n"
            )

        fig.suptitle(
            (
                f"{os.path.basename(path_to_snap)}\n"
                f"Aperture = {args.aperture_percent / 100:.2f} $R_{{500}}$\t\t"
                f"$z = {sw_data.metadata.z:.2f}$\tAge = {Cosmology().age(sw_data.metadata.z).value:.2f} Gyr\n"
                f"{z_agn_recent_text:s}"
                f"Central FoF group only"
            ),
            fontsize=7
        )

        if not args.quiet:
            fig.set_tight_layout(False)
            plt.show()

        fig.savefig(
            os.path.join(
                default_output_directory,
                f"cooling_times_{os.path.basename(path_to_snap)[:-5].replace('.', 'p')}.png"
            ),
            dpi=300
        )

        plt.close()
