import sys, os
import numpy as np
from matplotlib import pyplot as plt
from unyt import Solar_Mass

sys.path.append("..")

from scaling_relations import EntropyProfiles
from register import find_files, set_mnras_stylesheet, xlargs
from literature import Sun2009, Pratt2010


def make_profile(axes, **kwargs):
    profile_obj = EntropyProfiles(max_radius_r500=2.5, **kwargs)

    label = (
        f"Weights: {profile_obj.weighting}, "
        f"$\mu$-avg: {profile_obj.simple_electron_number_density}, "
        f"shell_avg: {profile_obj.shell_average}"
    )
    print(label)

    radial_bin_centres, entropy_profile, K500 = profile_obj.process_single_halo(
        path_to_snap=snap, path_to_catalogue=cat
    )
    entropy_profile /= K500

    print(repr(radial_bin_centres))
    print(repr(entropy_profile))

    axes.plot(
        radial_bin_centres,
        entropy_profile,
        linestyle='-',
        linewidth=1,
        label=label
    )


snap, cat = find_files()

set_mnras_stylesheet()
fig = plt.figure(constrained_layout=True)
axes = fig.add_subplot()

make_profile(axes, weighting='mass', simple_electron_number_density=False, shell_average=False)

axes.set_xscale('log')
axes.set_yscale('log')

axes.axvline(0.15, color='k', linestyle='--', lw=0.5, zorder=0)
axes.set_ylabel(r'$K/K_{500}$')
axes.set_xlabel(r'$r/r_{500}$')
axes.set_ylim([5e-3, 20])
axes.set_xlim([5e-3, 2.5])

sun_observations = Sun2009()
r_r500, S_S500_50, S_S500_10, S_S500_90 = sun_observations.get_shortcut()

axes.errorbar(
    r_r500,
    S_S500_50,
    yerr=(
        S_S500_50 - S_S500_10,
        S_S500_90 - S_S500_50
    ),
    fmt='o',
    color='black',
    ecolor='lightgray',
    elinewidth=0.5,
    capsize=0,
    label=sun_observations.citation
)
# axes.plot(r_r500, S_S500_50, c='grey', )
#
# axes.fill_between(
#     r_r500,
#     S_S500_10,
#     S_S500_90,
#     color='grey', alpha=0.4, linewidth=0
# )


rexcess = Pratt2010(n_radial_bins=30)
bin_median, bin_perc16, bin_perc84 = rexcess.combine_entropy_profiles(
    m500_limits=(
        1e14 * Solar_Mass,
        5e14 * Solar_Mass
    ),
    k500_rescale=True
)

axes.errorbar(
    rexcess.radial_bins,
    S_S500_50,
    yerr=(
        bin_median - bin_perc16,
        bin_perc84 - bin_median
    ),
    fmt='s',
    color='black',
    ecolor='lightgray',
    elinewidth=0.5,
    capsize=0,
    label=rexcess.citation
)
#
# axes.fill_between(
#     rexcess.radial_bins,
#     bin_perc16,
#     bin_perc84,
#     color='lime',
#     alpha=0.4,
#     linewidth=0
# )
# axes.plot(rexcess.radial_bins, bin_median, c='lime', label=rexcess.citation)

r = np.array([0.01, 1])
k = 1.40 * r ** 1.1
axes.plot(r, k, c='grey', ls='--', label='VKB (2005)')

fig.suptitle(
    (
        f"{os.path.basename(xlargs.run_directory)}\n"
        f"Central FoF group only\t\tEstimator: {xlargs.mass_estimator}"
    ),
    fontsize=4

)

plt.legend()

if not xlargs.quiet:
    plt.show()

plt.close()
