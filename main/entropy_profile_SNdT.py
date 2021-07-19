import sys, os
import numpy as np
from matplotlib import pyplot as plt
from unyt import Solar_Mass

sys.path.append("..")

from scaling_relations import EntropyProfiles
from register import set_mnras_stylesheet, xlargs, calibration_zooms
from literature import Sun2009, Pratt2010


dir = '/cosma/home/dp004/dc-alta2/data6/xl-zooms/hydro'
name = '/L0300N0564_VR2915_+1res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth'
snap_number = 30


set_mnras_stylesheet()
fig = plt.figure(constrained_layout=True)
axes = fig.add_subplot()

for sn_model in ['_SNdT7', '', '_SNdT8']:
    snap = dir + name + sn_model + '/snapshots' + name + f'_{snap_number:04d}.hdf5'
    cat = dir + name + sn_model + '/stf/' + name + f'_{snap_number:04d}/' + name + f'_{snap_number:04d}.properties'

    profile_obj = EntropyProfiles(
        max_radius_r500=2.5,
        weighting='mass',
        simple_electron_number_density=False,
        shell_average=False
    )

    radial_bin_centres, entropy_profile, K500 = profile_obj.process_single_halo(
        path_to_snap=snap, path_to_catalogue=cat
    )
    entropy_profile /= K500
    print(sn_model)
    # print(repr(radial_bin_centres))
    # print(repr(entropy_profile))

    if not sn_model:
        sn_model = '_SNdT7.5'

    axes.plot(
        radial_bin_centres,
        entropy_profile,
        linestyle='-',
        linewidth=1,
        label=sn_model.strip('_')
    )

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

axes.errorbar(
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
axes.plot(r, k, c='grey', ls='--', label='VKB (2005)', zorder=0)

axes.text(
    0.025,
    0.025,
    (
        f"{' '.join(name.split('_')[1:3])}\n"
        f"z = {calibration_zooms.redshift_from_index(snap_number):.2f}"
    ),
    color="k",
    ha="left",
    va="bottom",
    alpha=0.8,
    transform=axes.transAxes,
)

plt.legend()

if not xlargs.quiet:
    plt.show()

plt.close()
