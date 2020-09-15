import matplotlib

matplotlib.use('Agg')

import swiftsimio as sw
import h5py
import numpy as np
import sys
from unyt import unyt_quantity, Mpc, Solar_Mass
from matplotlib import pyplot as plt

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

T_hot = 1e5
fbary=0.15741 # Cosmic baryon fraction

run_name = sys.argv[1]
run_directory = sys.argv[2]
snapshot_name = sys.argv[3]
output_path = sys.argv[4]

snapshot_filename = f"{run_directory}/{snapshot_name}"
velociraptor_properties = f"{run_directory}/stf/{snapshot_name}"

with h5py.File(velociraptor_properties, 'r') as vrcatalogue:
    M500 = unyt_quantity(vrcatalogue['SO_Mass_500_rhocrit'][0], Solar_Mass)
    R500 = unyt_quantity(vrcatalogue['SO_R_500_rhocrit'][0], Mpc)
    x_cop = unyt_quantity(vrcatalogue['Xcminpot'][0], Mpc)
    y_cop = unyt_quantity(vrcatalogue['Ycminpot'][0], Mpc)
    z_cop = unyt_quantity(vrcatalogue['Zcminpot'][0], Mpc)

# Construct spatial mask to feed into swiftsimio
mask = sw.mask(snapshot_filename)
region = [
    [x_cop - R500, x_cop + R500],
    [y_cop - R500, y_cop + R500],
    [z_cop - R500, z_cop + R500]
]
mask.constrain_spatial(region)
data = sw.load(snapshot_filename, mask=mask)
posGas = data.gas.coordinates
massGas = data.gas.masses
tempGas = data.gas.temperatures
r = np.sqrt(
    (posGas[:, 0] - x_cop) ** 2 +
    (posGas[:, 1] - y_cop) ** 2 +
    (posGas[:, 2] - z_cop) ** 2
)

index = np.where((r < R500) & (tempGas > T_hot))[0]
Mhot500c = np.sum(massGas[index])
thisfhot500c = Mhot500c / M500

# Sun et al. 2009
M500_Sun = np.array(
    [3.18, 4.85, 3.90, 1.48, 4.85, 5.28, 8.49, 10.3, 2.0, 7.9, 5.6, 12.9, 8.0, 14.1, 3.22, 14.9, 13.4, 6.9, 8.95, 8.8,
     8.3, 9.7, 7.9]) * 1.e3
f500_Sun = np.array(
    [0.097, 0.086, 0.068, 0.049, 0.069, 0.060, 0.076, 0.081, 0.108, 0.086, 0.056, 0.076, 0.075, 0.114, 0.074, 0.088,
     0.094, 0.094, 0.078, 0.099, 0.065, 0.090, 0.093])
Mgas500_Sun=M500_Sun*f500_Sun

fig, ax = plt.subplots()
ax.plot(M500_Sun,Mgas500_Sun,'o',color='gray')
ax.plot(M500,Mhot500c,'s',color='blue')
ax.xlabel(r'$M_{500{\rm c}}/10^{10}{\rm M}_{\odot}$')
ax.ylabel(r'$M_{{\rm gas},500{\rm c}}/10^{10}{\rm M}_{\odot}$')