import numpy as np
import h5py as h5
from matplotlib import pyplot as plt


def wrap(dx, box):
    result = dx
    index = np.where(dx > (0.5 * box))[0]
    result[index] -= box
    index = np.where(dx < (-0.5 * box))[0]
    result[index] += box
    return result


boxMpc = 300.
# Range of M200c masses to select haloes in
minMassSelectMsun = 1.e13
maxMassSelectMsun = 1.e14
# Number of haloes to select within each mass bin, and number of mass bins
numHaloesPerBin = 1
numBins_select = 3
# Isolation criteria (distance and mass limit)
minDistFac = 10.  # isolation distance criterion (multiples of R200c)
minMassFrac = 0.1  # isolation mass criterion (fraction of M200c)

# EAGLE-XL data path
dataPath = "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/"
# VR data path
vrPath = dataPath + "stf_swiftdm_3dfof_subhalo_0036/"
# Halo properties file
haloPropFile = vrPath + "stf_swiftdm_3dfof_subhalo_0036.VELOCIraptor.properties.0"

# Read in halo properties
with h5.File(haloPropFile, 'r') as f:
    M200c = f['/Mass_200crit'][:] * 1.e10  # Msun units
    R200c = f['/R_200crit'][:]
    structType = f['/Structuretype'][:]
    xPotMin = f['/Xcminpot'][:]
    yPotMin = f['/Ycminpot'][:]
    zPotMin = f['/Zcminpot'][:]

# Cut down main sample to field haloes above minimum mass
index = np.where((structType == 10) & (M200c > (minMassFrac * minMassSelectMsun)))[0]
numHaloes = index.size
M200c = M200c[index]
R200c = R200c[index]
xPotMin = xPotMin[index]
yPotMin = yPotMin[index]
zPotMin = zPotMin[index]
print("Number of haloes in sample =", numHaloes)

# Select primary haloes with structure type 10 (field haloes) and within mass range
index = np.where((M200c > minMassSelectMsun) & (M200c < maxMassSelectMsun))[0]
print("Primary sample selected in M200c mass range [min,max] =", minMassSelectMsun, maxMassSelectMsun)
numHaloesPrimary = index.size
print("Number of haloes in primary sample =", numHaloesPrimary)
M200c_primary = M200c[index]
R200c_primary = R200c[index]
xPotMin_primary = xPotMin[index]
yPotMin_primary = yPotMin[index]
zPotMin_primary = zPotMin[index]

selectFlag = np.zeros(numHaloesPrimary, dtype=np.bool)
for i in np.arange(numHaloesPrimary - 1):
    minDistMpc = minDistFac * R200c_primary[i]  # Halo pair separation should be no smaller than this
    minMassMsun = minMassFrac * M200c_primary[i]  # Neighbour masses must not be larger than this
    # print(minDistMpc,minMassMsun)
    dx = wrap(xPotMin - xPotMin_primary[i], boxMpc)
    dy = wrap(yPotMin - yPotMin_primary[i], boxMpc)
    dz = wrap(zPotMin - zPotMin_primary[i], boxMpc)
    # print(dx.min(),dx.max(),dy.min(),dy.max(),dz.min(),dz.max())
    dr2 = (dx ** 2) + (dy ** 2) + (dz ** 2)
    index = np.where((M200c > minMassMsun) & (dr2 < minDistMpc ** 2))[0]
    if (index.size == 1): selectFlag[i] = True

# Subset of haloes that satisfy isolation criterion
M200c_iso = M200c_primary[selectFlag]
R200c_iso = R200c_primary[selectFlag]
xPotMin_iso = xPotMin_primary[selectFlag]
yPotMin_iso = yPotMin_primary[selectFlag]
zPotMin_iso = zPotMin_primary[selectFlag]
numHaloes_iso = M200c_iso.size
print('Number of haloes satisfying isolation critertion =', numHaloes_iso)

# Now select haloes at random within each mass bins
numHaloes_select = numHaloesPerBin * numBins_select
minMass = np.log10(minMassSelectMsun)
maxMass = np.log10(maxMassSelectMsun)
dlogM = (maxMass - minMass) / float(numBins_select)

print("Number of halo mass bins =", numBins_select)
print("Mass bins: log(Mmin) log(Mmax) dlogM =", minMass, maxMass, dlogM)
print("Number of haloes to select within each mass bin =", numHaloesPerBin)

M200c_select = np.zeros(numHaloes_select)
R200c_select = np.zeros(numHaloes_select)
xPotMin_select = np.zeros(numHaloes_select)
yPotMin_select = np.zeros(numHaloes_select)
zPotMin_select = np.zeros(numHaloes_select)

haloCounter = 0
for i in np.arange(numBins_select):
    bin1 = minMass + float(i) * dlogM
    bin2 = bin1 + dlogM
    index = np.where((np.log10(M200c_iso) > bin1) & (np.log10(M200c_iso) <= bin2))[0]
    # Select haloes at random without replacement
    selectedHaloes = index[np.random.choice(index.size, size=numHaloesPerBin, replace=False)]

    M200c_select[haloCounter:haloCounter + numHaloesPerBin] = M200c_iso[selectedHaloes]
    R200c_select[haloCounter:haloCounter + numHaloesPerBin] = R200c_iso[selectedHaloes]
    xPotMin_select[haloCounter:haloCounter + numHaloesPerBin] = xPotMin_iso[selectedHaloes]
    yPotMin_select[haloCounter:haloCounter + numHaloesPerBin] = yPotMin_iso[selectedHaloes]
    zPotMin_select[haloCounter:haloCounter + numHaloesPerBin] = zPotMin_iso[selectedHaloes]
    print(i, haloCounter, bin1, bin2, index.size, selectedHaloes)
    haloCounter += numHaloesPerBin

# Print out sample
for i in np.arange(numHaloes_select):
    print(i, M200c_select[i] / 1.e13, R200c_select[i], xPotMin_select[i], yPotMin_select[i], zPotMin_select[i])

# Sanity check - find the closest halo that is more massive
print('Sanity check: 2nd last column should always be more than last column')
for i in np.arange(numHaloes_select):
    dx = wrap(xPotMin - xPotMin_select[i], boxMpc)
    dy = wrap(yPotMin - yPotMin_select[i], boxMpc)
    dz = wrap(zPotMin - zPotMin_select[i], boxMpc)
    dr2 = (dx ** 2) + (dy ** 2) + (dz ** 2)
    index = np.where((M200c > minMassFrac * M200c_select[i]) & (dr2 > 0.001))[0]
    print(i, index.size, np.sqrt(dr2[index].min()), minDistFac * R200c_select[i])

# Plot HMF for selected objects and total
plt.figure()
plt.yscale('log')
plt.hist(np.log10(M200c), bins=6, range=(13., 15.))
plt.hist(np.log10(M200c[selectFlag]), bins=6, range=(13., 15.))
plt.show()
