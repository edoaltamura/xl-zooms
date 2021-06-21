import sys
import os.path
from matplotlib import pyplot as plt
import numpy as np
from unyt import Solar_Mass
import traceback

sys.path.append("..")

from scaling_relations import EntropyFgasSpace, EntropyProfiles
from register import find_files, set_mnras_stylesheet, xlargs
from literature import Sun2009, Pratt2010

snap, cat = find_files()
kwargs = dict(path_to_snap=snap, path_to_catalogue=cat)
set_mnras_stylesheet()

try:

    gas_profile_obj = EntropyFgasSpace(max_radius_r500=1.)
    radial_bin_centres, cumulative_gas_mass_profile, cumulative_mass_profile, m500fb = gas_profile_obj.process_single_halo(
        **kwargs)
    entropy_profile_obj = EntropyProfiles(max_radius_r500=1)
    _, entropy_profile, K500 = entropy_profile_obj.process_single_halo(**kwargs)

    entropy_profile /= K500
    gas_fraction_enclosed = cumulative_gas_mass_profile / m500fb

    fig = plt.figure(figsize=(4, 4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.7)
    axes = gs.subplots()

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
    axes[0, 1].set_ylim([1e-2, 2])
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
    # axes[1, 0].set_ylim([1e-3, 1])
    axes[1, 0].set_xlim([0.01, 1])

    axes[1, 1].plot(
        radial_bin_centres,
        entropy_profile * gas_fraction_enclosed ** (2 / 3),
        linestyle='-',
        color='r',
        linewidth=1,
        alpha=1,
    )
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_ylabel(r'$(K/K_{500}) \times f_{\rm gas}(<r) ^ {2/3}$')
    axes[1, 1].set_xlabel(r'$r/r_{500}$')
    axes[1, 1].set_ylim([1e-4, 3])
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
        color='black',
        ecolor='lightgray',
        elinewidth=0.5,
        capsize=0,
        label=sun_observations.citation
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

    r = np.array([0.01, 1])
    k = 1.40 * r ** 1.1
    axes.plot(r, k, c='grey', ls='--', label='VKB (2005)')

    fig.suptitle(
        (
            f"{os.path.basename(xlargs.run_directory)}\n"
            f"Central FoF group only\t\tEstimator: {xlargs.mass_estimator}\n"
            f"Redshift = {gas_profile_obj.z:.3f}"
        ),
        fontsize=4
    )
    if not xlargs.quiet:
        plt.show()
    plt.close()


except Exception as e:
    print(f"Snap number {xlargs.snapshot_number:04d} could not be processed.", e, sep='\n')
    traceback.print_exc()
