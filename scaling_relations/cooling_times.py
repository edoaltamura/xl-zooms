import numpy as np
import h5py as h5
import scipy.interpolate as sci
from matplotlib import pyplot as plt
from unyt import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
from matplotlib.colors import LogNorm
from scipy import stats

from literature import Cosmology
from register import Zoom, args, cooling_table
from .halo_property import HaloProperty

mean_molecular_weight = 0.59
mean_atomic_weight_per_free_electron = 1.14
primordial_hydrogen_mass_fraction = 0.76


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if f < 1e-2 or f > 1e3:
        float_str = "{0:.0e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


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


def calculate_mean_cooling_times(data, use_heating: bool = False):
    tff = np.sqrt(3 * np.pi / (32 * G * data.gas.densities))

    data_cooling = get_cooling_rates()
    data_heating = get_heating_rates()

    cooling_rates = np.log10(np.power(10., data_cooling[0, :, :, :, -2]) + np.power(10., data_cooling[0, :, :, :, -1]))
    heating_rates = np.log10(np.power(10., data_heating[0, :, :, :, -2]) + np.power(10., data_heating[0, :, :, :, -1]))

    if use_heating:
        print('Net cooling rates: heating - cooling')
        net_rates = np.log10(np.abs(np.power(10., heating_rates) - np.power(10., cooling_rates)))
    else:
        print('Only cooling rates')
        net_rates = cooling_rates

    axis = get_axis_tables()
    nH_grid = axis[0]
    T_grid = axis[4]
    Z_grid = axis[2]

    f_net_rates = sci.RegularGridInterpolator((T_grid, Z_grid, nH_grid), net_rates, method="linear", bounds_error=False,
                                              fill_value=-30)

    hydrogen_fraction = data.gas.element_mass_fractions.hydrogen
    gas_nH = (data.gas.densities / mh * hydrogen_fraction).to(cm ** -3)
    log_gas_nH = np.log10(gas_nH)
    temperature = data.gas.temperatures
    log_gas_T = np.log10(temperature)
    log_gas_Z = np.log10(data.gas.metal_mass_fractions.value / 0.0133714)

    too_hot = len(np.where(temperature > 1e9)[0])
    print(f'Detected {too_hot:d} particles hotter than 10^9 K.')

    # construct the matrix that we input in the interpolator
    values_to_int = np.zeros((len(log_gas_T), 3))
    values_to_int[:, 0] = log_gas_T
    values_to_int[:, 1] = log_gas_Z
    values_to_int[:, 2] = log_gas_nH

    net_rates_found = f_net_rates(values_to_int)

    cooling_times = np.log10(3. / 2. * 1.38e-16) + log_gas_T - log_gas_nH - net_rates_found - np.log10(3.154e13)

    return cooling_times


def draw_cooling_contours(axes, density_bins, temperature_bins):
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
    log_gas_Z = np.ones_like(temperature_interps) * np.log10(0.0133714 / 3)

    # construct the matrix that we input in the interpolator
    values_to_int = np.zeros((len(log_gas_T), 3))
    values_to_int[:, 0] = log_gas_T
    values_to_int[:, 1] = log_gas_Z
    values_to_int[:, 2] = log_gas_nH

    net_rates_found = f_net_rates(values_to_int)

    cooling_time = np.log10(3. / 2. * 1.38e-16) + log_gas_T - log_gas_nH - net_rates_found - np.log10(3.154e13)
    cooling_time = cooling_time.reshape(_density_interps.shape)

    # Define entropy levels to plot
    levels = np.log10(np.array([1, 1e2, 1e3, 5e3, 1e4]))
    fmt = {value: f'${latex_float(10 ** value)}$ Myr' for value in levels}
    contours = axes.contour(
        _density_interps,
        _temperature_interps,
        cooling_time,
        levels,
        colors='green',
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
        colors='green',
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
        a_heat = sw_data.gas.last_agnfeedback_scale_factors

        if agn_time is None:

            if z_agn_start < 7.2 or z_agn_end > 0:

                index = np.where(
                    (sw_data.gas.radial_distances < aperture_fraction) &
                    (sw_data.gas.fofgroup_ids == 1) &
                    (a_heat > (1 / (z_agn_start + 1))) &
                    (a_heat < (1 / (z_agn_end + 1))) &
                    (cooling_times > 7)
                )[0]
            else:
                index = np.where(
                    (sw_data.gas.radial_distances < aperture_fraction) &
                    (sw_data.gas.fofgroup_ids == 1)
                )[0]

            number_density = (sw_data.gas.densities / mh).to('cm**-3').value[index] * primordial_hydrogen_mass_fraction
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
            number_density = (density / mh).to('cm**-3').value * primordial_hydrogen_mass_fraction
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
            number_density = (density / mh).to('cm**-3').value * primordial_hydrogen_mass_fraction
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

        print("Number of particles being plotted", len(x))

        # Set the limits of the figure.
        assert (x > 0).all(), f"Found negative value(s) in x: {x[x <= 0]}"
        assert (y > 0).all(), f"Found negative value(s) in y: {y[y <= 0]}"

        # density_bounds = [1e-6, 1e4]  # in nh/cm^3
        # temperature_bounds = [1e3, 1e10]  # in K
        density_bounds = [1e-6, 1]  # in nh/cm^3
        temperature_bounds = [1e6, 1e10]  # in K
        bins = 256

        # Make the norm object to define the image stretch
        density_bins = np.logspace(
            np.log10(density_bounds[0]), np.log10(density_bounds[1]), bins
        )
        temperature_bins = np.logspace(
            np.log10(temperature_bounds[0]), np.log10(temperature_bounds[1]), bins
        )

        fig = plt.figure(figsize=(5, 5))
        gs = fig.add_gridspec(2, 2, hspace=0.1, wspace=0.2)
        axes = gs.subplots(sharex='col', sharey='row')

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
            draw_cooling_contours(ax, contour_density_bins, contour_temperature_bins)

            # Star formation threshold
            ax.axvline(0.1, color='k', linestyle=':', lw=1, zorder=0)

        # PLOT ALL PARTICLES ===============================================
        H, density_edges, temperature_edges = np.histogram2d(
            x, y, bins=[density_bins, temperature_bins]
        )

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
        ticklab = cbar.ax.get_yticklabels()
        ticks = cbar.ax.get_yticks()
        for i, (t, l) in enumerate(zip(ticks, ticklab)):
            if t < 100:
                ticklab[i] = f'{int(t):d}'
            else:
                ticklab[i] = f'$10^{{{int(np.log10(t)):d}}}$'
        cbar.ax.set_yticklabels(ticklab)

        txt = AnchoredText("All particles", loc="upper right", pad=0.4, borderpad=0, prop={"fontsize": 8})
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
            ticklab = cbar.ax.get_yticklabels()
            ticks = cbar.ax.get_yticks()
            for i, (t, l) in enumerate(zip(ticks, ticklab)):
                if t < 100:
                    ticklab[i] = f'{int(t):d}'
                else:
                    ticklab[i] = f'$10^{{{int(np.log10(t)):d}}}$'
            cbar.ax.set_yticklabels(ticklab)

        # Heating temperatures
        axes[0, 1].axhline(10 ** 7.5, color='k', linestyle='--', lw=1, zorder=0)
        txt = AnchoredText("SNe heated only", loc="upper right", pad=0.4, borderpad=0, prop={"fontsize": 8})
        axes[0, 1].add_artist(txt)

        # PLOT AGN HEATED PARTICLES ===============================================
        H, density_edges, temperature_edges = np.histogram2d(
            x[(agn_flag & ~snii_flag)],
            y[(agn_flag & ~snii_flag)],
            bins=[density_bins, temperature_bins]
        )
        vmax = np.max(H) + 1
        mappable = axes[1, 1].pcolormesh(
            density_edges, temperature_edges, H.T,
            norm=LogNorm(vmin=1, vmax=vmax), cmap='Reds_r', alpha=0.6
        )
        divider = make_axes_locatable(axes[1, 1])
        cax = divider.append_axes("right", size="3%", pad=0.)
        cbar = plt.colorbar(mappable, ax=axes[1, 1], cax=cax)
        ticklab = cbar.ax.get_yticklabels()
        ticks = cbar.ax.get_yticks()
        for i, (t, l) in enumerate(zip(ticks, ticklab)):
            if t < 100:
                ticklab[i] = f'{int(t):d}'
            else:
                ticklab[i] = f'$10^{{{int(np.log10(t)):d}}}$'
        cbar.ax.set_yticklabels(ticklab)

        txt = AnchoredText("AGN heated only", loc="upper right", pad=0.4, borderpad=0, prop={"fontsize": 8})
        axes[1, 1].add_artist(txt)
        # Heating temperatures
        axes[1, 1].axhline(10 ** 8.5, color='k', linestyle='--', lw=1, zorder=0)

        # PLOT AGN+SN HEATED PARTICLES ===============================================
        H, density_edges, temperature_edges = np.histogram2d(
            x[(agn_flag & snii_flag)],
            y[(agn_flag & snii_flag)],
            bins=[density_bins, temperature_bins]
        )
        vmax = np.max(H) + 1
        mappable = axes[1, 0].pcolormesh(
            density_edges, temperature_edges, H.T,
            norm=LogNorm(vmin=1, vmax=vmax), cmap='Purples_r', alpha=0.6
        )
        divider = make_axes_locatable(axes[1, 0])
        cax = divider.append_axes("right", size="3%", pad=0.)
        cbar = plt.colorbar(mappable, ax=axes[1, 0], cax=cax)
        ticklab = cbar.ax.get_yticklabels()
        ticks = cbar.ax.get_yticks()
        for i, (t, l) in enumerate(zip(ticks, ticklab)):
            if t < 100:
                ticklab[i] = f'{int(t):d}'
            else:
                ticklab[i] = f'$10^{{{int(np.log10(t)):d}}}$'
        cbar.ax.set_yticklabels(ticklab)

        txt = AnchoredText("AGN and SNe heated", loc="upper right", pad=0.4, borderpad=0, prop={"fontsize": 8})
        axes[1, 0].add_artist(txt)
        # Heating temperatures
        axes[1, 0].axhline(10 ** 8.5, color='k', linestyle='--', lw=1, zorder=0)
        axes[1, 0].axhline(10 ** 7.5, color='k', linestyle='--', lw=1, zorder=0)

        fig.text(0.5, 0.04, r"Density [$n_H$ cm$^{-3}$]", ha='center')
        fig.text(0.04, 0.5, r"Temperature [K]", va='center', rotation='vertical')

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
                f"Aperture = {args.aperture_percent / 100:.2f} $R_{{500}}$\t\t"
                f"$z = {sw_data.metadata.z:.2f}$\n"
                f"{z_agn_recent_text:s}"
                f"Central FoF group only"
            ),
            fontsize=7
        )

        if not args.quiet:
            plt.show()
        # fig.savefig(
        #     f'{calibration_zooms.output_directory}/density_temperature_{args.redshift_index:04d}.png',
        #     dpi=300
        # )

        plt.close()
