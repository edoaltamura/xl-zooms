import sys
from matplotlib import pyplot as plt
from unyt import Solar_Mass

sys.path.append("..")

from scaling_relations import EntropyProfiles
from register import find_files, set_mnras_stylesheet, xlargs
from literature import Sun2009, Pratt2010


def make_profile(axes, **kwargs):
    profile_obj = EntropyProfiles(max_radius_r500=1.5, **kwargs)
    print(
        f"xray_weighting: {profile_obj.xray_weighting} mu-average: {profile_obj.simple_electron_number_density} shell_average: {profile_obj.shell_average}")
    radial_bin_centres, entropy_profile, K500 = profile_obj.process_single_halo(
        path_to_snap=snap, path_to_catalogue=cat
    )
    entropy_profile /= K500
    axes.plot(
        radial_bin_centres,
        entropy_profile,
        linestyle='-',
        linewidth=1,
        label=f"xray: {profile_obj.xray_weighting}, $\mu$-avg: {profile_obj.simple_electron_number_density}, shell_avg: {profile_obj.shell_average}"
    )



snap, cat = find_files()

set_mnras_stylesheet()
fig = plt.figure(constrained_layout=True)
axes = fig.add_subplot()

make_profile(axes, xray_weighting=False, simple_electron_number_density=False, shell_average=False)
make_profile(axes, xray_weighting=True, simple_electron_number_density=False, shell_average=False)


axes.set_xscale('log')
axes.set_yscale('log')

axes.axvline(0.15, color='k', linestyle='--', lw=0.5, zorder=0)
axes.set_ylabel(r'$K/K_{500}$')
axes.set_xlabel(r'$r/r_{500}$')
axes.set_ylim([5e-3, 20])
axes.set_xlim([5e-3, 2.5])

sun_observations = Sun2009()
r_r500, S_S500_50, S_S500_10, S_S500_90 = sun_observations.get_shortcut()

axes.fill_between(
    r_r500,
    S_S500_10,
    S_S500_90,
    color='grey', alpha=0.4, linewidth=0
)
axes.plot(r_r500, S_S500_50, c='grey', label=sun_observations.citation)

rexcess = Pratt2010()
bin_median, bin_perc16, bin_perc84 = rexcess.combine_entropy_profiles(
    m500_limits=(
        1e14 * Solar_Mass,
        5e14 * Solar_Mass
    ),
    k500_rescale=True
)
axes.fill_between(
    rexcess.radial_bins,
    bin_perc16,
    bin_perc84,
    color='aqua',
    alpha=0.4,
    linewidth=0
)
axes.plot(rexcess.radial_bins, bin_median, c='blue', label=rexcess.citation)

plt.legend()

if not xlargs.quiet:
    plt.show()

plt.close()

# print(
#     gf.process_catalogue()
# )
