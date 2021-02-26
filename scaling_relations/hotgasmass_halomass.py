# Plot scaling relations for EAGLE-XL tests
import sys
import os
import unyt
import numpy as np
from typing import Tuple
import h5py as h5
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Make the register backend visible to the script
sys.path.append("../zooms")
sys.path.append("../observational_data")
sys.path.append("../hydrostatic_estimates")

from register import zooms_register, Zoom, Tcut_halogas, name_list
from vikhlinin_hse import HydrostaticEstimator
import observational_data as obs
import scaling_utils as utils
import scaling_style as style

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

cosmology = obs.Observations().cosmo_model
fbary = cosmology.Ob0 / cosmology.Om0  # Cosmic baryon fraction

plot_observation_errorbars = False


def process_single_halo(
        path_to_snap: str,
        path_to_catalogue: str,
        hse_dataset: pd.Series = None,
) -> Tuple[unyt.unyt_quantity]:
    # Read in halo properties
    with h5.File(f'{path_to_catalogue}', 'r') as h5file:
        M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)

        # If the hot gas mass already computed by VR, use that instead of calculating it
        Mhot500c_key = "SO_Mass_gas_highT_1.000000_times_500.000000_rhocrit"

        if Mhot500c_key in h5file.keys() and hse_dataset is None:
            Mhot500c = unyt.unyt_quantity(h5file[f'/{Mhot500c_key}'][0] * 1.e10, unyt.Solar_Mass)

        else:
            R500c = unyt.unyt_quantity(h5file['/SO_R_500_rhocrit'][0], unyt.Mpc)
            XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc)
            YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc)
            ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc)

            # If no custom aperture, select R500c as default
            if hse_dataset:
                assert R500c.units == hse_dataset["R500hse"].units
                assert M500c.units == hse_dataset["M500hse"].units
                R500c = hse_dataset["R500hse"]
                M500c = hse_dataset["M500hse"]

            import swiftsimio as sw

            # Read in gas particles
            mask = sw.mask(path_to_snap, spatial_only=False)
            region = [[XPotMin - R500c, XPotMin + R500c],
                      [YPotMin - R500c, YPotMin + R500c],
                      [ZPotMin - R500c, ZPotMin + R500c]]
            mask.constrain_spatial(region)
            mask.constrain_mask(
                "gas", "temperatures",
                Tcut_halogas * mask.units.temperature,
                1.e12 * mask.units.temperature
            )
            data = sw.load(path_to_snap, mask=mask)
            posGas = data.gas.coordinates
            massGas = data.gas.masses

            # Select hot gas within sphere
            deltaX = posGas[:, 0] - XPotMin
            deltaY = posGas[:, 1] - YPotMin
            deltaZ = posGas[:, 2] - ZPotMin
            deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2)
            index = np.where(deltaR < R500c)[0]
            Mhot500c = np.sum(massGas[index])
            Mhot500c = Mhot500c.to(unyt.Solar_Mass)

    return M500c, Mhot500c, Mhot500c / M500c


@utils.set_scaling_relation_name(os.path.splitext(os.path.basename(__file__))[0])
@utils.set_output_names([
    'M_500crit',
    'Mhot500c',
    'fhot500c'
])
def _process_single_halo(zoom: Zoom):

    try:
        hse_catalogue = pd.read_pickle(f'{zooms_register[0].output_directory}/hse_massbias.pkl')
        hse_catalogue_names = hse_catalogue['Run name'].values.tolist()
        print(f"Looking for HSE results in the catalogue - zoom: {zoom.run_name}")
        if zoom.run_name in hse_catalogue_names:
            i = hse_catalogue_names.index(zoom.run_name)
            hse_entry = hse_catalogue.loc[i]

    except FileExistsError as error:
        raise FileExistsError(
            f"{error}\n\nPlease, consider first generating the HSE catalogue for better performance."
        )

    return process_single_halo(zoom.snapshot_file, zoom.catalog_file, hse_dataset=hse_entry)


def m_500_hotgas(results: pd.DataFrame, data_label: str = 'crit'):
    fig, ax = plt.subplots()
    legend_handles = []
    for i in range(len(results)):

        run_style = style.get_style_for_object(results.loc[i, "Run name"])
        if run_style['Legend handle'] not in legend_handles:
            legend_handles.append(run_style['Legend handle'])

        ax.scatter(
            results.loc[i, "M_500crit"],
            results.loc[i, "Mhot500c"],
            marker=run_style['Marker style'],
            c=run_style['Color'],
            s=run_style['Marker size'],
            alpha=1,
            edgecolors='none',
            zorder=5
        )

    # Build legends
    legend_sims = plt.legend(handles=legend_handles, loc=2, frameon=True, facecolor='w', edgecolor='none')
    ax.add_artist(legend_sims)

    # Display observational data
    observations_color = (0.65, 0.65, 0.65)
    handles = []

    Sun09 = obs.Sun09()
    ax.scatter(Sun09.M_500, Sun09.M_500gas,
               marker='D', s=5, alpha=1, color=observations_color, edgecolors='none', zorder=0)
    if plot_observation_errorbars:
        ax.errorbar(Sun09.M_500, Sun09.M_500gas, yerr=Sun09.M_500gas_error, xerr=Sun09.M_500_error,
                    ls='none', elinewidth=0.5,  color=observations_color, zorder=0)
    handles.append(
        Line2D([], [], color=observations_color, marker='D', markeredgecolor='none', linestyle='None', markersize=4,
               label=Sun09.citation)
    )
    del Sun09

    Lovisari15 = obs.Lovisari15()
    ax.scatter(Lovisari15.M_500, Lovisari15.M_gas500, marker='^', s=5, alpha=1,
               color=observations_color, edgecolors='none', zorder=0)
    handles.append(
        Line2D([], [], color=observations_color, marker='^', markeredgecolor='none', linestyle='None', markersize=4,
               label=Lovisari15.citation)
    )
    del Lovisari15

    Lin12 = obs.Lin12()
    ax.scatter(Lin12.M_500, Lin12.M_500gas,
               marker='v', s=5, alpha=1, color=observations_color, edgecolors='none', zorder=0)
    if plot_observation_errorbars:
        ax.errorbar(Lin12.M_500, Lin12.M_500gas, yerr=Lin12.M_500gas_error, xerr=Lin12.M_500_error,
                    ls='none', elinewidth=0.5, color=observations_color, zorder=0)
    handles.append(
        Line2D([], [], color=observations_color, marker='v', markeredgecolor='none', linestyle='None', markersize=4,
               label=Lin12.citation)
    )
    del Lin12

    Eckert16 = obs.Eckert16()
    ax.plot(Eckert16.M_500_fit, Eckert16.M_500gas_fit, color=observations_color, zorder=0)
    handles.append(
        Line2D([], [], color=observations_color, linestyle='-', label=Eckert16.citation)
    )
    del Eckert16

    Vikhlinin06 = obs.Vikhlinin06()
    ax.scatter(Vikhlinin06.M_500, Vikhlinin06.M_500gas,
               marker='>', s=5, alpha=1, color=observations_color, edgecolors='none', zorder=0)
    if plot_observation_errorbars:
        ax.errorbar(Vikhlinin06.M_500, Vikhlinin06.M_500gas, yerr=Vikhlinin06.error_M_500gas, xerr=Vikhlinin06.error_M_500,
                    ls='none', elinewidth=0.5, color=observations_color, zorder=0)
    handles.append(
        Line2D([], [], color=observations_color, marker='>', markeredgecolor='none', linestyle='None', markersize=4,
               label=Vikhlinin06.citation)
    )
    del Vikhlinin06

    Barnes17 = obs.Barnes17().hdf5.z000p101.true
    relaxed = Barnes17.Ekin_500 / Barnes17.Ethm_500
    ax.scatter(Barnes17.M500[relaxed < 0.1], Barnes17.Mgas_500[relaxed < 0.1],
               marker='s', s=6, alpha=1, color='k', edgecolors='none', zorder=0)
    ax.scatter(Barnes17.M500[relaxed > 0.1], Barnes17.Mgas_500[relaxed > 0.1],
               marker='s', s=5, alpha=1, facecolors='w', edgecolors='k', linewidth=0.4, zorder=0)
    handles.append(
        Line2D([], [], color='k', marker='s', markeredgecolor='none', linestyle='None', markersize=4,
               label=obs.Barnes17().citation + ' $z=0.1$')
    )
    del Barnes17

    handles.append(
        Line2D([], [], color='black', linestyle='--', markersize=0, label=f"Planck18 $f_{{bary}}=${fbary:.3f}")
    )
    legend_obs = plt.legend(handles=handles, loc=4, frameon=True, facecolor='w', edgecolor='none')
    ax.add_artist(legend_obs)

    ax.set_xlabel(f'$M_{{500{{\\rm {data_label}}}\\ [{{\\rm M}}_{{\\odot}}]$')
    ax.set_ylabel(f'$M_{{500{{\\rm gas {data_label}}}\\ [{{\\rm M}}_{{\\odot}}]$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(ax.get_xlim(), [lim * fbary for lim in ax.get_xlim()], '--', color='k')

    fig.savefig(f'{zooms_register[0].output_directory}/m500{data_label}_hotgas.png', dpi=300)
    plt.show()
    plt.close()


def f_500_hotgas(results: pd.DataFrame, data_label: str = 'crit'):
    fig, ax = plt.subplots()
    legend_handles = []
    for i in range(len(results)):

        run_style = style.get_style_for_object(results.loc[i, "Run name"])
        if run_style['Legend handle'] not in legend_handles:
            legend_handles.append(run_style['Legend handle'])

        ax.scatter(
            results.loc[i, "M_500crit"],
            results.loc[i, "fhot500c"],
            marker=run_style['Marker style'],
            c=run_style['Color'],
            s=run_style['Marker size'],
            alpha=1,
            edgecolors='none',
            zorder=5
        )

    # Build legends
    legend_sims = plt.legend(handles=legend_handles, loc=2, frameon=True, facecolor='w', edgecolor='none')
    ax.add_artist(legend_sims)

    # Display observational data
    observations_color = (0.65, 0.65, 0.65)
    handles = []

    Sun09 = obs.Sun09()
    ax.scatter(Sun09.M_500, Sun09.fb_500,
               marker='D', s=5, alpha=1, color=observations_color, edgecolors='none', zorder=0)
    if plot_observation_errorbars:
        ax.errorbar(Sun09.M_500, Sun09.fb_500, yerr=Sun09.fb_500_error, xerr=Sun09.M_500_error,
                    ls='none', elinewidth=0.5, color=observations_color, zorder=0)
    handles.append(
        Line2D([], [], color=observations_color, marker='D', markeredgecolor='none', linestyle='None', markersize=4,
               label=Sun09.citation)
    )
    del Sun09

    Lovisari15 = obs.Lovisari15()
    ax.scatter(Lovisari15.M_500, Lovisari15.fb_500, marker='^', s=5, alpha=1,
               color=observations_color, edgecolors='none', zorder=0)
    handles.append(
        Line2D([], [], color=observations_color, marker='^', markeredgecolor='none', linestyle='None', markersize=4,
               label=Lovisari15.citation)
    )
    del Lovisari15

    Lin12 = obs.Lin12()
    ax.scatter(Lin12.M_500, Lin12.fb_500,
               marker='v', s=5, alpha=1, color=observations_color, edgecolors='none', zorder=0)
    if plot_observation_errorbars:
        ax.errorbar(Lin12.M_500, Lin12.fb_500, yerr=Lin12.fb_500_error, xerr=Lin12.M_500_error,
                    ls='none', elinewidth=0.5, color=observations_color, zorder=0)
    handles.append(
        Line2D([], [], color=observations_color, marker='v', markeredgecolor='none', linestyle='None', markersize=4,
               label=Lin12.citation)
    )
    del Lin12

    Eckert16 = obs.Eckert16()
    ax.plot(Eckert16.M_500_fit, Eckert16.fgas_500_fit, color=observations_color, zorder=0)
    handles.append(
        Line2D([], [], color=observations_color, linestyle='-', label=Eckert16.citation)
    )
    del Eckert16

    Vikhlinin06 = obs.Vikhlinin06()
    ax.scatter(Vikhlinin06.M_500, Vikhlinin06.fb_500,
               marker='>', s=5, alpha=1, color=observations_color, edgecolors='none', zorder=0)
    if plot_observation_errorbars:
        ax.errorbar(Vikhlinin06.M_500, Vikhlinin06.fb_500, yerr=Vikhlinin06.error_fb_500, xerr=Vikhlinin06.error_M_500,
                    ls='none', elinewidth=0.5, color=observations_color, zorder=0)
    handles.append(
        Line2D([], [], color=observations_color, marker='>', markeredgecolor='none', linestyle='None', markersize=4,
               label=Vikhlinin06.citation)
    )
    del Vikhlinin06

    Barnes17 = obs.Barnes17().hdf5.z000p101.true
    relaxed = Barnes17.Ekin_500 / Barnes17.Ethm_500
    ax.scatter(Barnes17.M500[relaxed < 0.1], Barnes17.Mgas_500[relaxed < 0.1] / Barnes17.M500[relaxed < 0.1],
               marker='s', s=6, alpha=1, color='k', edgecolors='none', zorder=0)
    ax.scatter(Barnes17.M500[relaxed > 0.1], Barnes17.Mgas_500[relaxed > 0.1] / Barnes17.M500[relaxed > 0.1],
               marker='s', s=5, alpha=1, facecolors='w', edgecolors='k', linewidth=0.4, zorder=0)
    handles.append(
        Line2D([], [], color='k', marker='s', markeredgecolor='none', linestyle='None', markersize=4,
               label=obs.Barnes17().citation + ' $z=0.1$')
    )
    del Barnes17

    handles.append(
        Line2D([], [], color='black', linestyle='--', markersize=0, label=f"Planck18 $f_{{bary}}=${fbary:.3f}")
    )
    legend_obs = plt.legend(handles=handles, loc=4, frameon=True, facecolor='w', edgecolor='none')
    ax.add_artist(legend_obs)

    ax.set_xlabel(f'$M_{{500{{\\rm {data_label}}}\\ [{{\\rm M}}_{{\\odot}}]$')
    ax.set_ylabel(f'$f_{{500{{\\rm gas, {data_label}}} / M_{{500{{\\rm {data_label}}}$')
    ax.set_xscale('log')
    ax.set_ylim([-0.07, 0.27])
    ax.set_xlim([4e12, 6e15])
    ax.plot(ax.get_xlim(), [fbary for _ in ax.get_xlim()], '--', color='k')

    fig.savefig(f'{zooms_register[0].output_directory}/f500{data_label}_hotgas.png', dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    import sys
    if sys.argv[1]:
        keyword = sys.argv[1]
    else:
        keyword = 'Ref'

    results = utils.process_catalogue(_process_single_halo, find_keyword=keyword)
    m_500_hotgas(results, data_label='hse')
    f_500_hotgas(results, data_label='hse')
