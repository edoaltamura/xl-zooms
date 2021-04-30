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


class EntropyComparison(HaloProperty):

    def __init__(self):
        super().__init__()

    def process_single_halo(
            self,
            zoom_obj: Zoom = None,
            path_to_snap: str = None,
            path_to_catalogue: str = None,
            agn_time: str = 'after',
            z_agn_start: float = 0.1,
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

        gamma = 5 / 3
        a_heat = sw_data.gas.last_agnfeedback_scale_factors

        index = np.where(
            (sw_data.gas.radial_distances < aperture_fraction) &
            (sw_data.gas.fofgroup_ids == 1) &
            (a_heat > (1 / (z_agn_start + 1))) &
            (a_heat < (1 / (z_agn_end + 1)))
        )[0]

        electron_number_density = (sw_data.gas.densities / mh / mean_molecular_weight).to('cm**-3')[index]
        temperature = sw_data.gas.temperatures.to('K')[index]
        entropy_snapshot = boltzmann_constant * temperature / electron_number_density ** (2 / 3)
        entropy_snapshot = entropy_snapshot.to('keV*cm**2').value

        # Entropy of all particles in aperture
        index = np.where(
            (sw_data.gas.radial_distances < aperture_fraction) &
            (sw_data.gas.fofgroup_ids == 1) &
            (sw_data.gas.temperatures > 1e5)
        )[0]

        electron_number_density = (sw_data.gas.densities / mh / mean_molecular_weight).to('cm**-3')[index]
        temperature = sw_data.gas.temperatures.to('K')[index]
        entropy_snapshot_all = boltzmann_constant * temperature / electron_number_density ** (2 / 3)
        entropy_snapshot_all = entropy_snapshot_all.to('keV*cm**2').value


        if agn_time == 'before':

            index = np.where(
                (sw_data.gas.radial_distances < aperture_fraction) &
                (sw_data.gas.fofgroup_ids == 1) &
                (a_heat > (1 / (z_agn_start + 1))) &
                (a_heat < (1 / (z_agn_end + 1))) &
                (sw_data.gas.densities_before_last_agnevent > 0)
            )[0]

            density = sw_data.gas.densities_before_last_agnevent[index]
            electron_number_density = (density / mh / mean_molecular_weight).to('cm**-3')
            A = sw_data.gas.entropies_before_last_agnevent[index] * sw_data.units.mass
            temperature = mean_molecular_weight * (gamma - 1) * (A * density ** (5 / 3 - 1)) / (
                    gamma - 1) * mh / boltzmann_constant
            temperature = temperature.to('K')

            entropy_heat = boltzmann_constant * temperature / electron_number_density ** (2 / 3)
            entropy_heat = entropy_heat.to('keV*cm**2').value

        elif agn_time == 'after':

            index = np.where(
                (sw_data.gas.radial_distances < aperture_fraction) &
                (sw_data.gas.fofgroup_ids == 1) &
                (a_heat > (1 / (z_agn_start + 1))) &
                (a_heat < (1 / (z_agn_end + 1))) &
                (sw_data.gas.densities_at_last_agnevent > 0)
            )[0]

            density = sw_data.gas.densities_at_last_agnevent[index]
            electron_number_density = (density / mh / mean_molecular_weight).to('cm**-3')
            A = sw_data.gas.entropies_at_last_agnevent[index] * sw_data.units.mass
            temperature = mean_molecular_weight * (gamma - 1) * (A * density ** (5 / 3 - 1)) / (
                    gamma - 1) * mh / boltzmann_constant
            temperature = temperature.to('K')

            entropy_heat = boltzmann_constant * temperature / electron_number_density ** (2 / 3)
            entropy_heat = entropy_heat.to('keV*cm**2').value

        agn_flag = sw_data.gas.heated_by_agnfeedback[index]
        snii_flag = sw_data.gas.heated_by_sniifeedback[index]
        agn_flag = agn_flag > 0
        snii_flag = snii_flag > 0

        x = entropy_snapshot
        y = entropy_heat
        z = sw_data.metadata.z

        print("Number of particles being plotted", len(x))

        # Set the limits of the figure.
        assert (x > 0).all(), f"Found negative value(s) in x: {x[x <= 0]}"
        assert (y > 0).all(), f"Found negative value(s) in y: {y[y <= 0]}"

        entropy_bounds = [1e-4, 1e6]  # in keV*cm**2
        bins = 256

        # Make the norm object to define the image stretch
        entropy_bins = np.logspace(
            np.log10(entropy_bounds[0]), np.log10(entropy_bounds[1]), bins
        )

        fig = plt.figure(figsize=(5, 5))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.45)
        axes = gs.subplots()

        for ax in axes.flat:
            ax.loglog()

            # Draw cross-hair marker
            T500 = (G * mean_molecular_weight * m500 * mp / r500 / 2 / boltzmann_constant).to('K').value
            K500 = (T500 * K * boltzmann_constant / (3 * m500 * Cosmology().fb / (4 * np.pi * r500 ** 3 * mp)) ** (
                    2 / 3)).to('keV*cm**2')

        # PLOT ALL PARTICLES ===============================================
        H, density_edges, temperature_edges = np.histogram2d(
            x, y, bins=[entropy_bins, entropy_bins]
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
        axes[0, 0].axvline(K500, color='k', linestyle=':', lw=1, zorder=0)
        axes[0, 0].axhline(K500, color='k', linestyle=':', lw=1, zorder=0)

        # PLOT SN HEATED PARTICLES ===============================================
        hist_bins = np.logspace(
            np.log10(np.min(np.r_[x, y, entropy_snapshot_all])),
            np.log10(np.max(np.r_[x, y, entropy_snapshot_all])),
            100
        )
        axes[0, 1].hist(entropy_snapshot_all, bins=hist_bins, histtype='step', label=f'All hot gas at z={z:.2f}')
        axes[0, 1].hist(x, bins=hist_bins, histtype='step', label=f'Heated gas at z={z:.2f}')
        axes[0, 1].hist(y, bins=hist_bins, histtype='step', label='Heated gas at feedback time')
        axes[0, 1].axvline(K500, color='k', linestyle=':', lw=1, zorder=0)
        axes[0, 1].set_xlabel(f"Entropy ({agn_time:s} heating) [keV cm$^2$]")
        axes[0, 1].set_ylabel('Number of particles')
        axes[0, 1].legend()

        # PLOT AGN HEATED PARTICLES ===============================================
        H, density_edges, temperature_edges = np.histogram2d(
            x[agn_flag],
            y[agn_flag],
            bins=[entropy_bins, entropy_bins]
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
        print(ticklab, ticks)
        for i, (t, l) in enumerate(zip(ticks, ticklab)):
            if t < 100:
                ticklab[i] = f'{int(t):d}'
            else:
                ticklab[i] = f'$10^{{{int(np.log10(t)):d}}}$'
        cbar.ax.set_yticklabels(ticklab)

        txt = AnchoredText("AGN heated only", loc="upper right", pad=0.4, borderpad=0, prop={"fontsize": 8})
        axes[1, 1].add_artist(txt)
        axes[1, 1].axvline(K500, color='k', linestyle=':', lw=1, zorder=0)
        axes[1, 1].axhline(K500, color='k', linestyle=':', lw=1, zorder=0)

        # PLOT AGN+SN HEATED PARTICLES ===============================================
        H, density_edges, temperature_edges = np.histogram2d(
            x[(agn_flag & snii_flag)],
            y[(agn_flag & snii_flag)],
            bins=[entropy_bins, entropy_bins]
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
        axes[1, 0].axvline(K500, color='k', linestyle=':', lw=1, zorder=0)
        axes[1, 0].axhline(K500, color='k', linestyle=':', lw=1, zorder=0)

        fig.text(0.5, 0.04,
                 f"Entropy (z = {calibration_zooms.redshift_from_index(args.redshift_index):.2f}) [keV cm$^2$]",
                 ha='center')
        fig.text(0.04, 0.5, f"Entropy ({agn_time:s} heating) [keV cm$^2$]", va='center', rotation='vertical')

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
                f"$z = {calibration_zooms.redshift_from_index(args.redshift_index):.2f}$\n"
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
