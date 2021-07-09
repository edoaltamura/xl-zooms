import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

sys.path.append("..")

from scaling_relations import VRProperties
from register import set_mnras_stylesheet, xlargs
set_mnras_stylesheet()

catalogue = VRProperties().process_catalogue()

catalogue['specific_sfr'] = catalogue['sfr_100kpc'] / catalogue['m_star100kpc']

fig = plt.figure(constrained_layout=True)
axes = fig.add_subplot()
axes.loglog()
axes.set_xlabel('Stellar mass $M_*$ [M$_\odot$]')
axes.set_ylabel(r"Specific SFR (100 kpc) [Gyr$^{-1}$]")

colors = list('rgbkc')
counter = 0
handles = []

for model in ['_-8res_', '_+1res_']:
    df = catalogue[catalogue['Run_name'].str.contains(model, regex=False)]
    kwargs = dict(s=5, color=colors[counter], edgecolors='none')
    axes.scatter(df['m_star100kpc'], df['specific_sfr'], **kwargs)
    handles.append(Line2D(
        [], [], color=colors[counter], marker='.', markeredgecolor='none', linestyle='None', markersize=4, label=model
    ))
    counter += 1


# axes.set_xlim(1.02, 0.07)
# axes.set_ylim(1.8e-4, 1.7)

axes.legend(handles=handles, frameon=True, facecolor='w', edgecolor='none', bbox_to_anchor=(0., 1.02, 1., .102),
           loc='lower left', mode="expand", borderaxespad=0.)

if not xlargs.quiet:
    plt.show()

plt.savefig('m_star100kpc__specific_sfr.pdf')

plt.close()

fig = plt.figure(constrained_layout=True)
axes = fig.add_subplot()
axes.loglog()
axes.set_xlabel('Stellar mass $M_*$ [M$_\odot$]')
axes.set_ylabel(r"Specific SFR (100 kpc) [Gyr$^{-1}$]")

colors = list('rgbkc')
counter = 0
handles = []

for model in ['_-8res_', '_+1res_']:
    df = catalogue[catalogue['Run_name'].str.contains(model, regex=False)]
    kwargs = dict(s=5, color=colors[counter], edgecolors='none')
    axes.scatter(df['m200'], df['specific_sfr'], **kwargs)
    handles.append(Line2D(
        [], [], color=colors[counter], marker='.', markeredgecolor='none', linestyle='None', markersize=4, label=model
    ))
    counter += 1


# axes.set_xlim(1.02, 0.07)
# axes.set_ylim(1.8e-4, 1.7)

axes.legend(handles=handles, frameon=True, facecolor='w', edgecolor='none', bbox_to_anchor=(0., 1.02, 1., .102),
           loc='lower left', mode="expand", borderaxespad=0.)

if not xlargs.quiet:
    plt.show()

plt.savefig('m200__specific_sfr.pdf')

plt.close()