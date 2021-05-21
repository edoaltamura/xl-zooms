from merge_catalogues import catalogue, select_runs, models
from matplotlib import pyplot as plt
from unyt import kb
from matplotlib.lines import Line2D

import sys

sys.path.append("..")

from literature import Sun2009
from register import args, calibration_zooms

try:
    plt.style.use("../register/mnras.mplstyle")
except:
    pass

fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(2, 3, hspace=0.2, wspace=0.)
axes = gs.subplots(sharex=True, sharey=True)

for ax in axes.flat:
    ax.loglog()

# 'Run_name', 'm_gas', 'f_gas', 'T500', 'T2500', 'T500_nocore',
#        'T2500_nocore', 'r2500', 'r1000', 'r500', 'r200', 'm2500', 'm1000',
#        'm500', 'm200', 'm_star500c', 'm_star30kpc', 'm_star', 'f_star',
#        'k30kpc', 'k2500', 'k1500', 'k1000', 'k500', 'k200', 'Ekin', 'Etherm',
#        'r_1500', 'm_1500', 'LX500', 'LX2500', 'LX500_nocore', 'LX2500_nocore'

# ['-8res_MinimumDistance_fixedAGNdT9.5_Nheat1_SNnobirth',
#  '-8res_Isotropic_fixedAGNdT8_Nheat1_SNnobirth',
#  '+1res_MinimumDistance_fixedAGNdT9_Nheat1_SNnobirth',
#  '+1res_Isotropic_fixedAGNdT8_Nheat1_SNnobirth',
#  '-8res_Isotropic_fixedAGNdT8.5_Nheat1_SNnobirth',
#  '+1res_Isotropic_fixedAGNdT9_Nheat1_SNnobirth',
#  '-8res_MinimumDistance_fixedAGNdT9_Nheat1_SNnobirth',
#  '+1res_MinimumDistance_fixedAGNdT8_Nheat1_SNnobirth',
#  '-8res_Ref',
#  '-8res_Isotropic_fixedAGNdT9_Nheat1_SNnobirth',
#  '-8res_MinimumDistance_fixedAGNdT7.5_Nheat1_SNnobirth',
#  '+1res_Isotropic_fixedAGNdT8.5_Nheat1_SNnobirth',
#  '-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth',
#  '-8res_Isotropic',
#  '+1res_Isotropic',
#  '+1res_MinimumDistance',
#  '-8res_MinimumDistance_fixedAGNdT8_Nheat1_SNnobirth',
#  '+1res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth',
#  '-8res_MinimumDistance']

axes[0, 0].set_ylabel('Entropy [keV cm$^2$]')
axes[0, 0].set_xlabel('$k_BT_{500}^{>0.15 r_{500}}$ [keV]')
axes[0, 1].set_xlabel('$k_BT_{500}^{>0.15 r_{500}}$ [keV]')
axes[0, 2].set_xlabel('$k_BT_{500}^{>0.15 r_{500}}$ [keV]')

axes[1, 0].set_ylabel('Entropy [keV cm$^2$]')
axes[1, 0].set_xlabel('$k_BT_{2500}^{>0.15 r_{500}}$ [keV]')
axes[1, 1].set_xlabel('$k_BT_{2500}^{>0.15 r_{500}}$ [keV]')
axes[1, 2].set_xlabel('$k_BT_{2500}^{>0.15 r_{500}}$ [keV]')

sun2009 = Sun2009()
kwargs = dict(color='k', marker='*', edgecolors='none', alpha=0.6)
axes[0, 0].scatter(sun2009.T_500, sun2009.K_500, **kwargs)
axes[0, 1].scatter(sun2009.T_500, sun2009.K_1000, **kwargs)
axes[0, 2].scatter(sun2009.T_500, sun2009.K_1500, **kwargs)
axes[1, 0].scatter(sun2009.T_2500, sun2009.K_2500, **kwargs)
axes[1, 1].scatter(sun2009.T_2500, sun2009.K_0p15r500, **kwargs)
axes[1, 2].scatter(sun2009.T_2500, sun2009.K_30kpc, **kwargs)

kwargs = dict(horizontalalignment='left', verticalalignment='top')
axes[0, 0].text(0.03, 0.97, r'$K_{500}$', transform=axes[0, 0].transAxes, **kwargs)
axes[0, 1].text(0.03, 0.97, r'$K_{1000}$', transform=axes[0, 1].transAxes, **kwargs)
axes[0, 2].text(0.03, 0.97, r'$K_{1500}$', transform=axes[0, 2].transAxes, **kwargs)
axes[1, 0].text(0.03, 0.97, r'$K_{2500}$', transform=axes[1, 0].transAxes, **kwargs)
axes[1, 1].text(0.03, 0.97, r'$K_{0.15r500}$', transform=axes[1, 1].transAxes, **kwargs)
axes[1, 2].text(0.03, 0.97, r'$K_{30 \rm kpc}$', transform=axes[1, 2].transAxes, **kwargs)

colors = list('rgbkc')
counter = 0
handles = []
for model in models:

    for keyword in args.keywords:

        if keyword in model:
            df = catalogue[catalogue['Run_name'].str.contains(model, regex=False)]

            kwargs = dict(s=5, color=colors[counter], edgecolors='none')

            axes[0, 0].scatter([(t * kb).to('keV') for t in df['T500_nocore']], df['k500'], **kwargs)
            axes[0, 1].scatter([(t * kb).to('keV') for t in df['T500_nocore']], df['k1000'], **kwargs)
            axes[0, 2].scatter([(t * kb).to('keV') for t in df['T500_nocore']], df['k1500'], **kwargs)
            axes[1, 0].scatter([(t * kb).to('keV') for t in df['T2500_nocore']], df['k2500'], **kwargs)
            axes[1, 1].scatter([(t * kb).to('keV') for t in df['T2500_nocore']], df['k0p15r500'], **kwargs)
            axes[1, 2].scatter([(t * kb).to('keV') for t in df['T2500_nocore']], df['k30kpc'], **kwargs)

            handles.append(Line2D([], [], color=colors[counter], marker='.', markeredgecolor='none', linestyle='None',
                                  markersize=4, label=model))

            counter += 1

handles.append(Line2D([], [], color='k', marker='*', markeredgecolor='none', linestyle='None', markersize=4,
                      label=sun2009.citation))
axes[0, 0].legend(handles=handles, frameon=True, facecolor='w', edgecolor='none', bbox_to_anchor=(0., 1.02, 1., .102),
           loc='lower left', mode="expand", borderaxespad=0.)

# fig.suptitle(
#     (
#         f"z = {calibration_zooms.redshift_from_index(args.redshift_index):.2f}\t\t (snap {args.redshift_index:04d})"
#     ),
#     fontsize=7
# )

plt.show()
