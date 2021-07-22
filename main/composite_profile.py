import sys, os
import numpy as np
from matplotlib import pyplot as plt
from unyt import Solar_Mass

sys.path.append("..")

from scaling_relations import (
    EntropyProfiles,
    TemperatureProfiles,
    PressureProfiles,
    DensityProfiles,
    IronProfiles,
    GasFractionProfiles
)
from register import find_files, set_mnras_stylesheet, xlargs
from literature import Sun2009, Pratt2010

snap, cat = find_files()

label = ' '.join(os.path.basename(snap).split('_')[1:3])

set_mnras_stylesheet()

fig = plt.figure(figsize=(6, 6), constrained_layout=True)
gs = fig.add_gridspec(2, 3, hspace=0.2, wspace=0.)
axes_all = gs.subplots(sharex=True, sharey=False)

for ax in axes_all.flat:
    ax.loglog()
    ax.axvline(1, color='k', linestyle='--', lw=0.5, zorder=0)

# ===================== Entropy
axes = axes_all[0, 0]
print('Entropy')
profile_obj = EntropyProfiles(max_radius_r500=2.5, weighting='mass', simple_electron_number_density=False,
                              shell_average=False)
radial_bin_centres, entropy_profile, K500 = profile_obj.process_single_halo(
    path_to_snap=snap, path_to_catalogue=cat
)
redshift = profile_obj.z
entropy_profile /= K500
axes.plot(
    radial_bin_centres,
    entropy_profile,
    linestyle='-',
    linewidth=1,
)

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
    markersize=1,
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
    markersize=1,
    label=rexcess.citation,
    zorder=0
)

r = np.array([0.01, 1])
k = 1.40 * r ** 1.1
axes.plot(r, k, c='grey', ls='--', label='VKB (2005)', zorder=0)
axes.legend()

# ===================== Temperature
axes = axes_all[0, 1]
print('Temperature')

profile_obj = TemperatureProfiles(max_radius_r500=2.5, weighting='mass', simple_electron_number_density=False,
                                  shell_average=False)
radial_bin_centres, temperature_profile, kBT500 = profile_obj.process_single_halo(
    path_to_snap=snap, path_to_catalogue=cat
)
temperature_profile /= kBT500
axes.plot(
    radial_bin_centres,
    temperature_profile,
    linestyle='-',
    linewidth=1,
    label=f'{label} z = {redshift:.2f}'
)
axes.set_ylabel(r'MW-Temperature $T/T_{500}$')
axes.set_xlabel(r'$r/r_{500}$')
axes.legend()

# ===================== Pressure
axes = axes_all[0, 2]
print('Pressure')

profile_obj = PressureProfiles(max_radius_r500=2.5, weighting='mass', simple_electron_number_density=False,
                               shell_average=False)
radial_bin_centres, pressure_profile, P500 = profile_obj.process_single_halo(
    path_to_snap=snap, path_to_catalogue=cat
)
pressure_profile /= P500
pressure_profile *= radial_bin_centres ** 3
axes.plot(
    radial_bin_centres,
    pressure_profile,
    linestyle='-',
    linewidth=1,
)
axes.set_ylabel(r'Pressure $P/P_{500}$ $(r/r_{500})^3$')
axes.set_xlabel(r'$r/r_{500}$')

# ===================== Density
axes = axes_all[1, 0]
print('Density')

profile_obj = DensityProfiles(max_radius_r500=2.5)
radial_bin_centres, density_profile, rho_crit = profile_obj.process_single_halo(
    path_to_snap=snap, path_to_catalogue=cat
)
density_profile /= rho_crit
density_profile *= radial_bin_centres ** 2
axes.plot(
    radial_bin_centres,
    density_profile,
    linestyle='-',
    linewidth=1,
)
axes.set_ylabel(r'Density $\rho/\rho_{\rm crit}$ $(r/r_{500})^2$')
axes.set_xlabel(r'$r/r_{500}$')

# ===================== Iron
axes = axes_all[1, 1]
print('Iron')

profile_obj = IronProfiles(max_radius_r500=2.5)
radial_bin_centres, iron_profile, Zsun = profile_obj.process_single_halo(
    path_to_snap=snap, path_to_catalogue=cat
)
iron_profile /= Zsun
axes.plot(
    radial_bin_centres,
    iron_profile,
    linestyle='-',
    linewidth=1,
)
axes.set_ylabel(r'$Metallicity Z_{\rm Fe}/Z_{\odot}$')
axes.set_xlabel(r'$r_{2Dproj}/r_{500}$')

# Ghizzardi et al. 2021 (X-COP)
ghizzardi_r_r500 = np.array([0.0125, 0.0375, 0.0625, 0.1125, 0.1875, 0.2625, 0.3375, 0.4125, 0.4875, 0.6, 0.775, 0.9975])
ghizzardi_Fe_mean = np.array([0.578, 0.432, 0.371, 0.317, 0.276, 0.243, 0.236, 0.245, 0.252, 0.250, 0.240, 0.200])
ghizzardi_Fe_sigma = np.array([0.193, 0.103, 0.066, 0.052, 0.042, 0.046, 0.042, 0.054, 0.064, 0.053, 0.047, 0.076])

axes.errorbar(
    ghizzardi_r_r500,
    ghizzardi_Fe_mean,
    yerr=(
        ghizzardi_Fe_sigma,
        ghizzardi_Fe_sigma
    ),
    fmt='o',
    color='grey',
    ecolor='lightgray',
    elinewidth=0.5,
    capsize=0,
    markersize=1,
    label='Ghizzardi et al. (2021)',
    zorder=0
)
axes.legend()

# ===================== Gas Fraction
axes = axes_all[1, 2]
print('Gas Fraction')

profile_obj = GasFractionProfiles(max_radius_r500=2.5)
radial_bin_centres, gas_fraaction_profile, m500 = profile_obj.process_single_halo(
    path_to_snap=snap, path_to_catalogue=cat
)
gas_fraaction_profile /= m500
axes.plot(
    radial_bin_centres,
    gas_fraaction_profile,
    linestyle='-',
    linewidth=1,
)
axes.set_ylabel(r'Hot gas fraction $M_{\rm gas}(<r)/M_{500}$')
axes.set_xlabel(r'$r/r_{500}$')

if not xlargs.quiet:
    plt.show()

plt.close()
