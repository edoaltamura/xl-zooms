import sys
import os.path
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from unyt import Solar_Mass, kb

sys.path.append("..")

from scaling_relations import EntropyFgasSpace, EntropyProfiles, MWTemperatures
from register import set_mnras_stylesheet, xlargs, default_output_directory, calibration_zooms, delete_last_line
from literature import Sun2009, Pratt2010, Croston2008, Cosmology

set_mnras_stylesheet()


def myLogFormat(y, pos):
    # Find the number of decimal places required
    if y >= 1:
        decimalplaces = int(np.maximum(-np.log10(y), 0))  # =0 for numbers >=1
        # Insert that number into a format string
        formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    elif y >= 0.1 or y < 1:
        decimalplaces = int(np.maximum(-np.log10(y), 0))  # =0 for numbers >=1
        # Insert that number into a format string
        formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)


# Set axes limits
fgas_bounds = [10 ** (-2.5), 1]  # dimensionless
k_bounds = [1e-2, 7]  # K500 units
r_bounds = [0.01, 1]  # R500 units
t_bounds = [0.4, 5]  # keV units

fig = plt.figure(figsize=(4.5, 5), constrained_layout=True)
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.7, height_ratios=[0.05, 1, 1], width_ratios=[1, 1])
axes = gs.subplots()

cosmology = Cosmology()
redshift = calibration_zooms.redshift_from_index(xlargs.redshift_index)

temperature_obj = MWTemperatures()
temperatures_dataframe = temperature_obj.process_catalogue()
data_color = np.log10(np.array([(i * kb).to('keV').v for i in temperatures_dataframe['T0p75r500_nocore']]))
temperatures_dataframe['color'] = (data_color - np.log10(t_bounds[0])) / (np.log10(t_bounds[1]) - np.log10(t_bounds[0]))

gas_profile_obj = EntropyFgasSpace(max_radius_r500=1.)
gas_profiles_dataframe = gas_profile_obj.process_catalogue()

entropy_profile_obj = EntropyProfiles(max_radius_r500=1)
entropy_profiles_dataframe = entropy_profile_obj.process_catalogue()

# Merge all catalogues into one
catalogue = temperatures_dataframe
for datasets in [gas_profiles_dataframe, entropy_profiles_dataframe]:
    catalogue = pd.concat([catalogue, datasets], axis=1)

# Remove duplicate columns
catalogue = catalogue.loc[:, ~catalogue.columns.duplicated()]
print(catalogue.info())

for i in range(len(catalogue)):
    row = catalogue.loc[i]
    name = row["Run_name"]
    temperature = row['T0p75r500_nocore']
    color = plt.cm.jet(row['color'])
    radial_bin_centres = row['radial_bin_centres']
    gas_fraction_enclosed = row['cumulative_gas_mass_profile'] / row['m500fb']
    entropy_profile = row['k'] / row['k500']

    vr_number = name.split('_')[1][2:]
    print(f"Printing halo VR{vr_number}")

    line_style = dict(linestyle='-',
                      color=color,
                      linewidth=0.5,
                      alpha=0.6)

    axes[1, 0].plot(
        gas_fraction_enclosed,
        entropy_profile,
        **line_style
    )

    axes[1, 1].plot(
        radial_bin_centres,
        entropy_profile,
        **line_style
    )

    axes[2, 0].plot(
        radial_bin_centres,
        gas_fraction_enclosed,
        **line_style
    )
    axes[2, 1].plot(
        radial_bin_centres,
        entropy_profile * gas_fraction_enclosed ** (2 / 3) * cosmology.ez_function(redshift) ** (2 / 3),
        **line_style
    )

    delete_last_line()

axes[0, 0].remove()
axes[0, 1].remove()
top_row_axes = fig.add_subplot(gs[0, :])
cax = top_row_axes.inset_axes([0.25, 0, 0.5, 1], transform=top_row_axes.transAxes)
top_row_axes.axis('off')
colorbar = mpl.colorbar.ColorbarBase(
    cax,
    cmap=mpl.cm.jet,
    norm=mpl.colors.LogNorm(vmin=t_bounds[0], vmax=t_bounds[1]),
    orientation='horizontal'
)
colorbar.set_label(r'$k_B T_{\rm mw}(0.15-0.75~r_{500})$ [keV]')
colorbar.ax.xaxis.set_ticks_position('top')
colorbar.ax.xaxis.set_label_position('top')
colorbar.ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
colorbar.ax.xaxis.set_major_formatter('{x:.0f}')

axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('linear')
axes[1, 0].set_ylabel(r'$K/K_{500}$')
axes[1, 0].set_xlabel(r'$f_{\rm gas}(<r) = M_{\rm gas} / (M_{500}\ f_b)$')
axes[1, 0].set_ylim(k_bounds)
axes[1, 0].set_xlim(fgas_bounds)

axes[1, 1].set_xscale('log')
axes[1, 1].set_yscale('log')
axes[1, 1].set_ylabel(r'$K/K_{500}$')
axes[1, 1].set_xlabel(r'$r/r_{500}$')
axes[1, 1].set_ylim(k_bounds)
axes[1, 1].set_xlim(r_bounds)

axes[2, 0].set_xscale('log')
axes[2, 0].set_yscale('log')
axes[2, 0].set_ylabel(r'$f_{\rm gas}(<r) = M_{\rm gas} / (M_{500}\ f_b)$')
axes[2, 0].set_xlabel(r'$r/r_{500}$')
axes[2, 0].set_ylim(fgas_bounds)
axes[2, 0].set_xlim(r_bounds)

axes[2, 1].set_xscale('log')
axes[2, 1].set_yscale('log')
axes[2, 1].set_ylabel(r'$E(z) ^ {2/3}~(K/K_{500})~\times~f_{\rm gas}(<r) ^ {2/3}$')
axes[2, 1].set_xlabel(r'$r/r_{500}$')
axes[2, 1].set_ylim([k_bounds[0] * fgas_bounds[0] ** (2 / 3), k_bounds[1] * fgas_bounds[1] ** (2 / 3)])
axes[2, 1].set_xlim(r_bounds)

sun_observations = Sun2009()
r_r500, S_S500_50, S_S500_10, S_S500_90 = sun_observations.get_shortcut()

axes[1, 1].errorbar(
    r_r500,
    S_S500_50,
    yerr=(
        S_S500_50 - S_S500_10,
        S_S500_90 - S_S500_50
    ),
    fmt='o',
    color='grey',
    ecolor='lightgray',
    elinewidth=0.5,
    capsize=0,
    markersize=2,
    label=sun_observations.citation,
    zorder=0
)

rexcess = Pratt2010(n_radial_bins=30)
bin_median, bin_perc16, bin_perc84 = rexcess.combine_entropy_profiles(
    m500_limits=(
        1e14 * Solar_Mass,
        5e14 * Solar_Mass
    ),
    k500_rescale=True
)

axes[1, 1].errorbar(
    rexcess.radial_bins,
    bin_median,
    yerr=(
        bin_median - bin_perc16,
        bin_perc84 - bin_median
    ),
    fmt='s',
    color='grey',
    ecolor='lightgray',
    elinewidth=0.5,
    capsize=0,
    markersize=2,
    label=rexcess.citation,
    zorder=0
)

r = np.array([0.01, 1])
k = 1.40 * r ** 1.1
axes[1, 1].plot(r, k, c='grey', ls='--', label='VKB (2005)', zorder=0)
axes[1, 1].legend(loc="lower right")

croston = Croston2008()
croston.compute_gas_mass()
croston.estimate_total_mass()
croston.compute_gas_fraction()
for i, cluster in enumerate(croston.cluster_data):
    kwargs = dict(c='grey', alpha=0.7, lw=0.3)
    if i == 0:
        kwargs = dict(c='grey', alpha=0.4, lw=0.3, label=croston.citation)
    axes[2, 0].plot(cluster['r_r500'], cluster['f_g'] / cosmology.fb0, zorder=0, **kwargs)
axes[2, 0].legend(loc="upper left")

fig.suptitle(
    (
        f"{' '.join(xlargs.keywords)}\n"
        f"Central FoF group only\t\tEstimator: {xlargs.mass_estimator}\n"
        f"Redshift = {redshift:.3f}"
    ),
    fontsize=6
)
if not xlargs.quiet:
    plt.show()

fig.savefig(
    os.path.join(
        default_output_directory,
        f"entropy_vs_fgas_temp_coloured_{xlargs.redshift_index:04d}.png"
    ),
    dpi=300
)

plt.close()
