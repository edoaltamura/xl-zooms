import numpy as np
from unyt import unyt_quantity, kpc, Mpc, mh, K, boltzmann_constant, cm, G, mp
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
from matplotlib.colors import LogNorm

from .halo_property import HaloProperty
from register import Zoom, calibration_zooms, args
from literature import Cosmology

mean_molecular_weight = 0.59
mean_atomic_weight_per_free_electron = 1.14


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
    # temperature_interps *= unyt.K * unyt.boltzmann_constant
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
    )

    # work with logarithms for loglog scale
    # middle of the figure:
    # xmin, xmax, ymin, ymax = plt.axis()
    # logmid = (np.log10(xmin) + np.log10(xmax)) / 2, (np.log10(ymin) + np.log10(ymax)) / 2

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

            logmid = log_rho, np.log10(levels[i]) - 2 * log_rho / 3
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


class TemperatureDensity(HaloProperty):

    def __init__(self):
        super().__init__()

    def process_single_halo(
            self,
            zoom_obj: Zoom = None,
            path_to_snap: str = None,
            path_to_catalogue: str = None,
            agn_time: str = None,
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
        sw_data.gas.temperatures.convert_to_physical()
        sw_data.gas.densities.convert_to_physical()
        sw_data.gas.densities_before_last_agnevent.convert_to_physical()
        sw_data.gas.densities_at_last_agnevent.convert_to_physical()

        gamma = 5 / 3
        z_agn_recent = 0.5

        if agn_time is None:
            index = np.where((sw_data.gas.radial_distances < aperture_fraction) & (sw_data.gas.fofgroup_ids == 1))[
                0]
            number_density = (sw_data.gas.densities / mh).to('cm**-3').value[index]
            temperature = sw_data.gas.temperatures.to('K').value[index]

        elif agn_time == 'before':
            index = np.where(
                (sw_data.gas.radial_distances < aperture_fraction) &
                (sw_data.gas.fofgroup_ids == 1) &
                (sw_data.gas.densities_before_last_agnevent > 0) &
                (sw_data.gas.last_agnfeedback_scale_factors > (1 / (z_agn_recent + 1)))
            )[0]
            density = sw_data.gas.densities_before_last_agnevent[index]
            number_density = (density / mh).to('cm**-3').value
            A = sw_data.gas.entropies_before_last_agnevent[index] * sw_data.units.mass
            temperature = mean_molecular_weight * (gamma - 1) * (A * density ** (5 / 3 - 1)) / (
                        gamma - 1) * mh / boltzmann_constant
            temperature = temperature.to('K').value

        elif agn_time == 'after':
            index = np.where(
                (sw_data.gas.radial_distances < aperture_fraction) &
                (sw_data.gas.fofgroup_ids == 1) &
                (sw_data.gas.densities_at_last_agnevent > 0) &
                (sw_data.gas.last_agnfeedback_scale_factors > (1 / (z_agn_recent + 1)))
            )[0]
            density = sw_data.gas.densities_at_last_agnevent[index]
            number_density = (density / mh).to('cm**-3').value
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
        nH_500 = (rho_crit * 500 / mh).to('cm**-3')

        x = number_density
        y = temperature

        # Set the limits of the figure.
        assert (x > 0).all(), f"Found negative value(s) in x: {x[x <= 0]}"
        assert (y > 0).all(), f"Found negative value(s) in y: {y[y <= 0]}"

        density_bounds = [1e-6, 1e4]  # in nh/cm^3
        temperature_bounds = [1e3, 1e10]  # in K
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

            n_adiabats = np.array([1.e-4, 1.]) * nH_500.units
            T_adiabats = (K500 * n_adiabats ** (2 / 3) / boltzmann_constant).to('K')
            ax.plot(n_adiabats, T_adiabats, linewidth=1, color='r')

            draw_adiabats(ax, density_bins, temperature_bins)

            # Star formation threshold
            ax.axvline(0.1, color='k', linestyle=':', lw=1)

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
        plt.colorbar(mappable, ax=axes[0, 0], cax=cax)
        txt = AnchoredText("All particles", loc="upper right", pad=0.4, borderpad=0, prop={"fontsize": 8})
        axes[0, 0].add_artist(txt)

        # PLOT SN HEATED PARTICLES ===============================================
        H, density_edges, temperature_edges = np.histogram2d(
            x[(snii_flag & ~agn_flag)],
            y[(snii_flag & ~agn_flag)],
            bins=[density_bins, temperature_bins]
        )
        vmax = np.max(H) + 1
        mappable = axes[0, 1].pcolormesh(
            density_edges, temperature_edges, H.T,
            norm=LogNorm(vmin=1, vmax=vmax), cmap='Greens_r', alpha=0.6
        )
        divider = make_axes_locatable(axes[0, 1])
        cax = divider.append_axes("right", size="3%", pad=0.)
        plt.colorbar(mappable, ax=axes[0, 1], cax=cax)
        # Heating temperatures
        axes[0, 1].axhline(10 ** 7.5, color='k', linestyle='--', lw=1)
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
        plt.colorbar(mappable, ax=axes[1, 1], cax=cax)
        txt = AnchoredText("AGN heated only", loc="upper right", pad=0.4, borderpad=0, prop={"fontsize": 8})
        axes[1, 1].add_artist(txt)
        # Heating temperatures
        axes[1, 1].axhline(10 ** 8.5, color='k', linestyle='--', lw=1)

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
        plt.colorbar(mappable, ax=axes[1, 0], cax=cax)
        txt = AnchoredText("AGN and SNe heated", loc="upper right", pad=0.4, borderpad=0, prop={"fontsize": 8})
        axes[1, 0].add_artist(txt)
        # Heating temperatures
        axes[1, 0].axhline(10 ** 8.5, color='k', linestyle='--', lw=1)
        axes[1, 0].axhline(10 ** 7.5, color='k', linestyle='--', lw=1)

        fig.text(0.5, 0.04, r"Density [$n_H$ cm$^{-3}$]", ha='center')
        fig.text(0.04, 0.5, r"Temperature [K]", va='center', rotation='vertical')
        fig.suptitle(
            (
                f"Aperture = {args.aperture_percent / 100:.2f} $R_{{500}}$\t\t"
                f"$z = {calibration_zooms.redshift_from_index(args.redshift_index):.2f}$\n"
                f"Selecting gas heated since redshift: {z_agn_recent} (a = {1 / (z_agn_recent + 1):.2f})\n"
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
