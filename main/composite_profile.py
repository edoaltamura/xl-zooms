import sys, os
import numpy as np
from matplotlib import pyplot as plt
from unyt import Solar_Mass, mp

sys.path.append("..")

from scaling_relations import (
    EntropyProfiles,
    TemperatureProfiles,
    PressureProfiles,
    DensityProfiles,
    IronProfiles,
    GasFractionProfiles
)
from register import find_files, set_mnras_stylesheet, xlargs, mean_atomic_weight_per_free_electron
from literature import Sun2009, Pratt2010, Croston2008, Cosmology

snap, cat = find_files()

label = ' '.join(os.path.basename(snap).split('_')[1:3])

set_mnras_stylesheet()

cosmology = Cosmology()

fig = plt.figure(figsize=(6, 6), constrained_layout=True)
gs = fig.add_gridspec(2, 3, hspace=0., wspace=0.)
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
# axes.set_xlabel(r'$r/r_{500}$')
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
# axes.set_xlabel(r'$r/r_{500}$')
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

croston = Croston2008()
croston.compute_gas_mass()
croston.estimate_total_mass()
croston.compute_gas_fraction()
for i, cluster in enumerate(croston.cluster_data):
    kwargs = dict(c='grey', alpha=0.7, lw=0.3)
    if i == 0:
        kwargs = dict(c='grey', alpha=0.4, lw=0.3, label=croston.citation)
    axes.plot(cluster['r_r500'], cluster['n_e'] * mp / mean_atomic_weight_per_free_electron, zorder=0, **kwargs)

axes.legend()

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
axes.set_ylabel(r'Metallicity $Z_{\rm Fe}/Z_{\odot}$')
axes.set_xlabel(r'$r_{2Dproj}/r_{500}$')

# Observational data taken from compilation by Gastaldello et al. 2021 (Fig.4)
Palette_Blue = '#2CBDFE'
Palette_Green = '#47DBCD'
Palette_Pink = '#F3A0F2'
Palette_Purple = '#9D2EC5'
Palette_Violet = '#661D98'
Palette_Amber = '#F5B14C'

# Mernier et al. 2017 data
Mer17_rad_ave = np.array([0.004, 0.016, 0.033, 0.053, 0.082, 0.125, 0.205, 0.615])
Mer17_Fe_ave = np.array([0.839, 0.804, 0.708, 0.661, 0.541, 0.445, 0.341, 0.276])
Mer17_Fe_low = np.array([0.633, 0.67, 0.512, 0.48, 0.361, 0.311, 0.214, 0.133])
Mer17_Fe_high = np.array([1.045, 0.938, 0.903, 0.841, 0.721, 0.579, 0.468, 0.42])

axes.plot(Mer17_rad_ave, Mer17_Fe_ave, linestyle='-', color=Palette_Blue, label='Mernier et al. (2017)')
axes.plot(Mer17_rad_ave, Mer17_Fe_low, linestyle=':', color=Palette_Blue)
axes.plot(Mer17_rad_ave, Mer17_Fe_high, linestyle=':', color=Palette_Blue)
axes.fill_between(Mer17_rad_ave, Mer17_Fe_low, Mer17_Fe_high, alpha=0.2, color=Palette_Blue)

# Lovisari and Reiprich 2019
Lov19_rad_ave = np.array([0.007, 0.022, 0.043, 0.072, 0.112, 0.168, 0.25, 0.425, 1.025])
Lov19_Fe_ave = np.array([0.624, 0.578, 0.534, 0.472, 0.392, 0.309, 0.211, 0.226, 0.256])
Lov19_Fe_low = np.array([0.37, 0.316, 0.307, 0.278, 0.256, 0.212, 0.124, 0.154, 0.191])
Lov19_Fe_high = np.array([0.879, 0.84, 0.761, 0.667, 0.529, 0.406, 0.299, 0.298, 0.32])

axes.plot(Lov19_rad_ave, Lov19_Fe_ave, linestyle='-', color=Palette_Green, label='Lovisari et al. (2019)')
axes.plot(Lov19_rad_ave, Lov19_Fe_low, linestyle=':', color=Palette_Green)
axes.plot(Lov19_rad_ave, Lov19_Fe_high, linestyle=':', color=Palette_Green)
axes.fill_between(Lov19_rad_ave, Lov19_Fe_low, Lov19_Fe_high, alpha=0.2, color=Palette_Green)

# Ghizzardi et al. 2021
Ghi21_rad_ave = np.array([0.012, 0.038, 0.062, 0.112, 0.188, 0.262, 0.338, 0.412, 0.488, 0.60, 0.775, 0.998])
Ghi21_Fe_ave = np.array([0.855, 0.639, 0.549, 0.469, 0.408, 0.359, 0.349, 0.362, 0.373, 0.37, 0.355, 0.296])
Ghi21_Fe_low = np.array([0.569, 0.487, 0.453, 0.392, 0.346, 0.291, 0.287, 0.283, 0.278, 0.293, 0.284, 0.183])
Ghi21_Fe_high = np.array([1.142, 0.791, 0.646, 0.546, 0.47, 0.429, 0.411, 0.444, 0.467, 0.448, 0.425, 0.41])

axes.plot(Ghi21_rad_ave, Ghi21_Fe_ave, linestyle='-', color=Palette_Amber, label='Ghizzardi et al. (2021)')
axes.plot(Ghi21_rad_ave, Ghi21_Fe_low, linestyle=':', color=Palette_Amber)
axes.plot(Ghi21_rad_ave, Ghi21_Fe_high, linestyle=':', color=Palette_Amber)
axes.fill_between(Ghi21_rad_ave, Ghi21_Fe_low, Ghi21_Fe_high, alpha=0.2, color=Palette_Amber)

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
axes.axhline(cosmology.fb0, color='k', linestyle='--', lw=0.5, zorder=0, label='$f_b$ Universal baryon fraction')
axes.set_ylabel(r'Hot gas fraction $M_{\rm gas}(<r)/M_{500}$')
axes.set_xlabel(r'$r/r_{500}$')

croston = Croston2008()
croston.compute_gas_mass()
croston.estimate_total_mass()
croston.compute_gas_fraction()
for i, cluster in enumerate(croston.cluster_data):
    kwargs = dict(c='grey', alpha=0.7, lw=0.3)
    if i == 0:
        kwargs = dict(c='grey', alpha=0.4, lw=0.3, label=croston.citation)
    axes[2, 0].plot(cluster['r_r500'], cluster['f_g'], zorder=0, **kwargs)

axes.legend()

if not xlargs.quiet:
    plt.show()

plt.close()
