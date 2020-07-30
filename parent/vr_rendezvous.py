import numpy as np
import h5py as h5

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# GLOBAL PARAMETERS ##########################################################

sample_M200c = np.asarray([1.487, 3.033, 6.959]) * 1e13
sample_R200c = np.asarray([0.519, 0.658, 0.868])
sample_x = np.asarray([134.688, 90.671, 71.962])
sample_y = np.asarray([169.921, 289.822, 69.291])
sample_z = np.asarray([289.233, 98.227, 240.338])

# EAGLE-XL data path
dataPath = "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/"
# VR data path
vrPath = dataPath + "stf_swiftdm_3dfof_subhalo_0036/"
# Halo properties file
haloPropFile = vrPath + "stf_swiftdm_3dfof_subhalo_0036.VELOCIraptor.properties.0"

#############################################################################

# Read in halo properties
with h5.File(haloPropFile, 'r') as f:
    M200c = f['/Mass_200crit'][:] * 1.e10  # Msun units
    R200c = f['/R_200crit'][:]
    structType = f['/Structuretype'][:]
    xPotMin = f['/Xcminpot'][:]
    yPotMin = f['/Ycminpot'][:]
    zPotMin = f['/Zcminpot'][:]

print("Set-up search parameters")
# Cut down main sample to field haloes above minimum mass
index = np.where((structType == 10) & (M200c > 1e13))[0]
numHaloes = index.size
M200c = M200c[index]
R200c = R200c[index]
xPotMin = xPotMin[index]
yPotMin = yPotMin[index]
zPotMin = zPotMin[index]
print("Number of haloes in sample =", numHaloes)

with open("outfiles/halo_selected_SK.txt", "w") as text_file:
    print("# Halo counter, M200c/1.e13 [Msun], R200c [Mpc], xPotMin [Mpc], yPotMin [Mpc], zPotMin [Mpc]",
          file=text_file)

for i in range(3):
    print(f"Finding properties of halo {i:d}...")
    idx_find_M200c = find_nearest(M200c, sample_M200c[i])
    idx_find_R200c = find_nearest(R200c, sample_R200c[i])
    idx_find_x = find_nearest(xPotMin, sample_x[i])
    idx_find_y = find_nearest(yPotMin, sample_y[i])
    idx_find_z = find_nearest(zPotMin, sample_z[i])

    # Check if all search results are the same
    result = np.array([idx_find_M200c,
                       idx_find_R200c,
                       idx_find_x,
                       idx_find_y,
                       idx_find_z], dtype=np.int)

    if np.all(result == result[0]):
        idx_found = result[0]
        print(i, M200c[idx_found] / 1.e13, R200c[idx_found], xPotMin[idx_found], yPotMin[idx_found], zPotMin[idx_found])

        # Print to txt file
        with open("outfiles/halo_selected_SK.txt", "w") as text_file:
            print(
                f"{i}, {M200c[idx_found] / 1.e13}, {R200c[idx_found]}, {xPotMin[idx_found]}, {yPotMin[idx_found]}, {zPotMin[idx_found]}",
                file=text_file)