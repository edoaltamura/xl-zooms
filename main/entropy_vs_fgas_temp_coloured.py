import sys
import os.path
from matplotlib import pyplot as plt
import numpy as np
from unyt import Solar_Mass

sys.path.append("..")

from scaling_relations import EntropyFgasSpace, EntropyProfiles, MWTemperatures
from register import set_mnras_stylesheet, xlargs, default_output_directory
from literature import Sun2009, Pratt2010, Croston2008, Cosmology

set_mnras_stylesheet()

fig = plt.figure(figsize=(5, 5), constrained_layout=True)
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.7)
axes = gs.subplots()

cosmology = Cosmology()

gas_profile_obj = EntropyFgasSpace(max_radius_r500=1.)
gas_profiles_dataframe = gas_profile_obj.process_catalogue()
gas_fraction_enclosed = gas_profiles_dataframe['cumulative_gas_mass_profile'] / gas_profiles_dataframe['m500fb']
radial_bin_centres = gas_profiles_dataframe['radial_bin_centres']

print(radial_bin_centres)

entropy_profile_obj = EntropyProfiles(max_radius_r500=1)
entropy_profiles_dataframe = gas_profile_obj.process_catalogue()
print(entropy_profiles_dataframe)
entropy_profile = entropy_profiles_dataframe['entropy_profile'] / entropy_profiles_dataframe['K500']

temperature_obj = MWTemperatures()
temperatures_dataframe = temperature_obj.process_catalogue()
temperatures = temperatures_dataframe['T500_nocore']


axes[0, 0].plot(
    gas_fraction_enclosed,
    entropy_profile,
    linestyle='-',
    color='r',
    linewidth=1,
    alpha=1,
)
axes[0, 0].set_xscale('linear')
axes[0, 0].set_yscale('linear')
axes[0, 0].set_ylabel(r'$K/K_{500}$')
axes[0, 0].set_xlabel(r'$f_{\rm gas}(<r) = M_{\rm gas} / (M_{500}\ f_b)$')
axes[0, 0].set_ylim([0, 2])
axes[0, 0].set_xlim([0, 1])

axes[0, 1].plot(
    radial_bin_centres,
    entropy_profile,
    linestyle='-',
    color='r',
    linewidth=1,
    alpha=1,
)
axes[0, 1].set_xscale('log')
axes[0, 1].set_yscale('log')
axes[0, 1].set_ylabel(r'$K/K_{500}$')
axes[0, 1].set_xlabel(r'$r/r_{500}$')
axes[0, 1].set_ylim([1e-2, 5])
axes[0, 1].set_xlim([0.01, 1])

axes[1, 0].plot(
    radial_bin_centres,
    gas_fraction_enclosed,
    linestyle='-',
    color='r',
    linewidth=1,
    alpha=1,
)
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')
axes[1, 0].set_ylabel(r'$f_{\rm gas}(<r) = M_{\rm gas} / (M_{500}\ f_b)$')
axes[1, 0].set_xlabel(r'$r/r_{500}$')
axes[1, 0].set_ylim([1e-5, 1])
axes[1, 0].set_xlim([0.01, 1])

axes[1, 1].plot(
    radial_bin_centres,
    entropy_profile * gas_fraction_enclosed ** (2 / 3) * cosmology.ez_function(gas_profile_obj.z) ** (2 / 3),
    linestyle='-',
    color='r',
    linewidth=1,
    alpha=1,
)
axes[1, 1].set_xscale('log')
axes[1, 1].set_yscale('log')
axes[1, 1].set_ylabel(r'$E(z) ^ {2/3}~(K/K_{500})~\times~f_{\rm gas}(<r) ^ {2/3}$')
axes[1, 1].set_xlabel(r'$r/r_{500}$')
axes[1, 1].set_ylim([5e-5, 3])
axes[1, 1].set_xlim([0.01, 1])

sun_observations = Sun2009()
r_r500, S_S500_50, S_S500_10, S_S500_90 = sun_observations.get_shortcut()

axes[0, 1].errorbar(
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

axes[0, 1].errorbar(
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
axes[0, 1].plot(r, k, c='grey', ls='--', label='VKB (2005)', zorder=0)
axes[0, 1].legend(loc="lower right")

croston = Croston2008()
croston.compute_gas_mass()
croston.estimate_total_mass()
croston.compute_gas_fraction()
for i, cluster in enumerate(croston.cluster_data):
    kwargs = dict(c='grey', alpha=0.7, lw=0.3)
    if i == 0:
        kwargs = dict(c='grey', alpha=0.4, lw=0.3, label=croston.citation)
    axes[1, 0].plot(cluster['r_r500'], cluster['f_g'] / gas_profile_obj.fb, zorder=0, **kwargs)
axes[1, 0].legend(loc="upper left")

fig.suptitle(
    (
        f"{os.path.basename(xlargs.run_directory)}\n"
        f"Central FoF group only\t\tEstimator: {xlargs.mass_estimator}\n"
        f"Redshift = {gas_profile_obj.z:.3f}"
    ),
    fontsize=6
)
if not xlargs.quiet:
    plt.show()

# fig.savefig(
#     os.path.join(
#         default_output_directory,
#         f"entropy_vs_fgas.png"
#     ),
#     dpi=300
# )

plt.close()

