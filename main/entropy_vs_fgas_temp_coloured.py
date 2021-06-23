import sys
import os.path
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from unyt import Solar_Mass

sys.path.append("..")

from scaling_relations import EntropyFgasSpace, EntropyProfiles, MWTemperatures
from register import set_mnras_stylesheet, xlargs, default_output_directory, calibration_zooms
from literature import Sun2009, Pratt2010, Croston2008, Cosmology

set_mnras_stylesheet()

fig = plt.figure(figsize=(5, 5), constrained_layout=True)
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.7, height_ratios=[0.05, 1, 1], width_ratios=[1, 1])
axes = gs.subplots()

cosmology = Cosmology()
redshift = calibration_zooms.redshift_from_index(xlargs.redshift_index)

temperature_obj = MWTemperatures()
temperatures_dataframe = temperature_obj.process_catalogue()
data_color = np.log10(np.array([i.v for i in temperatures_dataframe['T500_nocore']]))
temperatures_dataframe['color'] = (data_color - np.min(data_color)) / (np.max(data_color) - np.min(data_color))

gas_profile_obj = EntropyFgasSpace(max_radius_r500=1.)
gas_profiles_dataframe = gas_profile_obj.process_catalogue()

entropy_profile_obj = EntropyProfiles(max_radius_r500=1)
entropy_profiles_dataframe = entropy_profile_obj.process_catalogue()

catalogue = temperatures_dataframe
for datasets in [gas_profiles_dataframe, entropy_profiles_dataframe]:
    catalogue = pd.concat(
        [
            catalogue,
            datasets
        ],
        axis=1
    )

# Remove duplicate columns
catalogue = catalogue.loc[:, ~catalogue.columns.duplicated()]
print(catalogue.info())

alpha = 0.5

for i in range(len(catalogue)):
    row = catalogue.loc[i]
    name = row["Run_name"]
    temperature = row['T500_nocore']
    color = plt.cm.jet(row['color'])
    radial_bin_centres = row['radial_bin_centres']
    gas_fraction_enclosed = row['cumulative_gas_mass_profile'] / row['m500fb']
    entropy_profile = row['k'] / row['k500']

    vr_number = name.split('_')[1][2:]
    print(f"Printing halo VR{vr_number}")

    axes[1, 0].plot(
        gas_fraction_enclosed,
        entropy_profile,
        linestyle='-',
        color=color,
        linewidth=1,
        alpha=alpha,
    )

    axes[1, 1].plot(
        radial_bin_centres,
        entropy_profile,
        linestyle='-',
        color=color,
        linewidth=1,
        alpha=alpha,
    )

    axes[2, 0].plot(
        radial_bin_centres,
        gas_fraction_enclosed,
        linestyle='-',
        color=color,
        linewidth=1,
        alpha=alpha,
    )
    axes[2, 1].plot(
        radial_bin_centres,
        entropy_profile * gas_fraction_enclosed ** (2 / 3) * cosmology.ez_function(redshift) ** (2 / 3),
        linestyle='-',
        color=color,
        linewidth=1,
        alpha=alpha,
    )

norm = mpl.colors.LogNorm(
    vmin=temperatures_dataframe['T500_nocore'].min().v,
    vmax=temperatures_dataframe['T500_nocore'].max().v
)
axes[0, 0].remove()
axes[0, 1].remove()
top_row_axes = fig.add_subplot(gs[0, :])
cax = top_row_axes.inset_axes([0.25, 0, 0.5, 1], transform=top_row_axes.transAxes)
top_row_axes.axis('off')
colorbar = mpl.colorbar.ColorbarBase(
    cax,
    cmap=mpl.cm.jet,
    norm=norm,
    orientation='horizontal'
)
colorbar.set_label(r'$T_{500}^{\rm core~excised}$ [K]')
colorbar.ax.xaxis.set_ticks_position('top')
colorbar.ax.xaxis.set_label_position('top')

axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('linear')
axes[1, 0].set_ylabel(r'$K/K_{500}$')
axes[1, 0].set_xlabel(r'$f_{\rm gas}(<r) = M_{\rm gas} / (M_{500}\ f_b)$')
axes[1, 0].set_ylim([0, 5])
axes[1, 0].set_xlim([10 ** (-2.5), 1])

axes[1, 1].set_xscale('log')
axes[1, 1].set_yscale('log')
axes[1, 1].set_ylabel(r'$K/K_{500}$')
axes[1, 1].set_xlabel(r'$r/r_{500}$')
axes[1, 1].set_ylim([1e-2, 5])
axes[1, 1].set_xlim([0.01, 1])

axes[2, 0].set_xscale('log')
axes[2, 0].set_yscale('log')
axes[2, 0].set_ylabel(r'$f_{\rm gas}(<r) = M_{\rm gas} / (M_{500}\ f_b)$')
axes[2, 0].set_xlabel(r'$r/r_{500}$')
axes[2, 0].set_ylim([10 ** (-2.5), 1])
axes[2, 0].set_xlim([0.01, 1])

axes[2, 1].set_xscale('log')
axes[2, 1].set_yscale('log')
axes[2, 1].set_ylabel(r'$E(z) ^ {2/3}~(K/K_{500})~\times~f_{\rm gas}(<r) ^ {2/3}$')
axes[2, 1].set_xlabel(r'$r/r_{500}$')
axes[2, 1].set_ylim([5e-5, 3])
axes[2, 1].set_xlim([0.01, 1])

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
        f"entropy_vs_fgas.png"
    ),
    dpi=300
)

plt.close()
