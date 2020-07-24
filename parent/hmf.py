import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

volMpc3 = 300. ** 3

# EAGLE-XL data path
dataPath = "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/"
# VR data path
vrPath = dataPath + "stf_swiftdm_3dfof_subhalo_0036/"

# Halo properties file
haloPropFile = vrPath + "stf_swiftdm_3dfof_subhalo_0036.VELOCIraptor.properties.0"

# Read in halo properties
h5file = h5.File(haloPropFile, 'r')
h5dset = h5file['/Mass_200crit']
M200c = h5dset[...]
h5file.close()
# Convert to Msun units
M200c *= 1.e10

# Plot histogram of HMF
numBins = 12
minBin = 13.
maxBin = 16.
dlogX = (maxBin - minBin) / float(numBins)
binX = minBin + dlogX * (0.5 + np.arange(numBins))

plt.figure()
plt.xlim([minBin, maxBin])
plt.yscale('log')
# hist,edges=np.histogram(np.log10(M200c),bins=numBins,range=(minBin,maxBin))
plt.hist(np.log10(M200c), bins=numBins, range=(minBin, maxBin))
# binY=hist#/(volMpc3*binX)
# plt.plot(binX,binY)
plt.show()
