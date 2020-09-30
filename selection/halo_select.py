import sys
import numpy as np
import h5py as h5
import yaml
import swiftsimio


def wrap(dx, box):
    result = dx
    index = np.where(dx > (0.5 * box))[0]
    result[index] -= box
    index = np.where(dx < (-0.5 * box))[0]
    result[index] += box
    return result


# Load parameter file
with open(sys.argv[1], "r") as handle:
    params = yaml.load(handle, Loader=yaml.Loader)

np.random.seed(int(params['random_seed']))
massChoice = bool(params['mass_choice'])
minMassSelectMsun = 10. ** float(params['min_mass_select_log10'])
maxMassSelectMsun = 10. ** float(params['max_mass_select_log10'])
numHaloesPerBin = int(params['num_halos_per_bin'])
numBins_select = int(params['num_bins_select'])
minDistFac = float(params['min_dist_fac'])
minDistMpc = float(params['min_dist_mpc'])
minMassFrac = float(params['min_mass_frac'])

# Load box size
snap = swiftsimio.load(params['snap_file'])
boxMpc = snap.metadata.boxsize[0]
del snap

# Read in halo properties
with h5.File(params['vr_file'], 'r') as h5file:
    if massChoice == 0:
        print('Using M200c/R200c')
        MDeltac = h5file['/Mass_200crit'][:] * 1.e10  # Msun units
        RDeltac = h5file['/R_200crit'][:]
    else:
        print('Using M500c/R500c')
        MDeltac = h5file['/SO_Mass_500_rhocrit'][:] * 1.e10  # Msun units
        RDeltac = h5file['/SO_R_500_rhocrit'][:]

    structType = h5file['/Structuretype'][:]
    xPotMin = h5file['/Xcminpot'][:]
    yPotMin = h5file['/Ycminpot'][:]
    zPotMin = h5file['/Zcminpot'][:]

# Store position in list (0 is most massive)
index = np.argsort(MDeltac)[::-1]  # most massive to least massive
MDeltac = MDeltac[index]
RDeltac = RDeltac[index]
xPotMin = xPotMin[index]
yPotMin = yPotMin[index]
zPotMin = zPotMin[index]
indexList = np.arange(MDeltac.size)

# Cut down main sample to field haloes above minimum mass
index = np.where((structType == 10) & (MDeltac > (minMassFrac * minMassSelectMsun)))[0]
numHaloes = index.size
MDeltac = MDeltac[index]
RDeltac = RDeltac[index]
xPotMin = xPotMin[index]
yPotMin = yPotMin[index]
zPotMin = zPotMin[index]
indexList = indexList[index]
print("Number of haloes in sample =", numHaloes)

# Select primary haloes with structure type 10 (field haloes) and within mass range
index = np.where((MDeltac > minMassSelectMsun) & (MDeltac < maxMassSelectMsun))[0]
print("Primary sample selected in MDeltac mass range [min,max] =", minMassSelectMsun, maxMassSelectMsun)
numHaloesPrimary = index.size
print("Number of haloes in primary sample =", numHaloesPrimary)
MDeltac_primary = MDeltac[index]
RDeltac_primary = RDeltac[index]
xPotMin_primary = xPotMin[index]
yPotMin_primary = yPotMin[index]
zPotMin_primary = zPotMin[index]
indexList_primary = indexList[index]

selectFlag = np.zeros(numHaloesPrimary, dtype=np.bool)
for i in np.arange(numHaloesPrimary - 1):
    # Halo pair separation should be no smaller than this
    minDist = np.max([minDistMpc, minDistFac * RDeltac_primary[i]])
    # Neighbour masses must not be larger than this
    minMassMsun = minMassFrac * MDeltac_primary[i]
    dx = wrap(xPotMin - xPotMin_primary[i], boxMpc)
    dy = wrap(yPotMin - yPotMin_primary[i], boxMpc)
    dz = wrap(zPotMin - zPotMin_primary[i], boxMpc)
    dr2 = (dx ** 2) + (dy ** 2) + (dz ** 2)
    index = np.where((MDeltac > minMassMsun) & (dr2 < minDist ** 2))[0]
    if index.size == 1:
        selectFlag[i] = True

# Subset of haloes that satisfy isolation criterion
MDeltac_iso = MDeltac_primary[selectFlag]
RDeltac_iso = RDeltac_primary[selectFlag]
xPotMin_iso = xPotMin_primary[selectFlag]
yPotMin_iso = yPotMin_primary[selectFlag]
zPotMin_iso = zPotMin_primary[selectFlag]
indexList_iso = indexList_primary[selectFlag]
numHaloes_iso = MDeltac_iso.size
print('Number of haloes satisfying isolation critertion =', numHaloes_iso)

# Now select haloes at random within each mass bins
numHaloes_select = numHaloesPerBin * numBins_select
minMass = np.log10(minMassSelectMsun)
maxMass = np.log10(maxMassSelectMsun)
dlogM = (maxMass - minMass) / float(numBins_select)

print("Number of halo mass bins =", numBins_select)
print("Mass bins: log(Mmin) log(Mmax) dlogM =", minMass, maxMass, dlogM)
print("Number of haloes to select within each mass bin =", numHaloesPerBin)

# Save mass bin info
mass_bins_repository = dict()
bin_counter = 0
for i in np.arange(numBins_select):
    bin1 = minMass + float(i) * dlogM
    bin2 = bin1 + dlogM
    index_this_bin = np.where((np.log10(MDeltac_iso) > bin1) & (np.log10(MDeltac_iso) <= bin2))[0]
    mass_bins_repository[f'mass_bin{bin_counter}'] = dict()
    mass_bins_repository[f'mass_bin{bin_counter}']['mass_min_log'] = float(bin1)
    mass_bins_repository[f'mass_bin{bin_counter}']['mass_max_log'] = float(bin2)
    mass_bins_repository[f'mass_bin{bin_counter}']['num_halos'] = len(indexList_iso[index_this_bin])
    mass_bins_repository[f'mass_bin{bin_counter}']['index_list'] = indexList_iso[index_this_bin].tolist()
    mass_bins_repository[f'mass_bin{bin_counter}']['mass_list'] = MDeltac_iso[index_this_bin].tolist()

    print(f'mass_bins_repository INFO: mass_bin{bin_counter}')
    for key in mass_bins_repository[f'mass_bin{bin_counter}']:
        if type(mass_bins_repository[f'mass_bin{bin_counter}'][key]) != list:
            print(f"\t{key:<13s}: {mass_bins_repository[f'mass_bin{bin_counter}'][key]}")
        else:
            print(
                f"\t{key:<13s}: "
                f"[{mass_bins_repository[f'mass_bin{bin_counter}'][key][0]:d} ... "
                f"{mass_bins_repository[f'mass_bin{bin_counter}'][key][-1]:d}]"
            )

    bin_counter += 1

with open(f"{params['output_dir']}/mass_bins_repository.yml", "w") as handle:
    yaml.dump(mass_bins_repository, handle, default_flow_style=False)

# Initialise arrays for random selection from each bin
MDeltac_select = np.zeros(numHaloes_select)
RDeltac_select = np.zeros(numHaloes_select)
xPotMin_select = np.zeros(numHaloes_select)
yPotMin_select = np.zeros(numHaloes_select)
zPotMin_select = np.zeros(numHaloes_select)
indexList_select = np.zeros(numHaloes_select, dtype=np.int)

haloCounter = 0
for i in np.arange(numBins_select):
    bin1 = minMass + float(i) * dlogM
    bin2 = bin1 + dlogM
    index = np.where((np.log10(MDeltac_iso) > bin1) & (np.log10(MDeltac_iso) <= bin2))[0]

    # Select haloes at random without replacement
    selectedHaloes = index[np.random.choice(index.size, size=numHaloesPerBin, replace=False)]

    MDeltac_select[haloCounter:haloCounter + numHaloesPerBin] = MDeltac_iso[selectedHaloes]
    RDeltac_select[haloCounter:haloCounter + numHaloesPerBin] = RDeltac_iso[selectedHaloes]
    xPotMin_select[haloCounter:haloCounter + numHaloesPerBin] = xPotMin_iso[selectedHaloes]
    yPotMin_select[haloCounter:haloCounter + numHaloesPerBin] = yPotMin_iso[selectedHaloes]
    zPotMin_select[haloCounter:haloCounter + numHaloesPerBin] = zPotMin_iso[selectedHaloes]
    indexList_select[haloCounter:haloCounter + numHaloesPerBin] = indexList_iso[selectedHaloes]
    print(i, haloCounter, bin1, bin2, index.size, selectedHaloes)
    haloCounter += numHaloesPerBin

# Sort sample in mass order
index = np.argsort(MDeltac_select)
MDeltac_select = MDeltac_select[index]
RDeltac_select = RDeltac_select[index]
xPotMin_select = xPotMin_select[index]
yPotMin_select = yPotMin_select[index]
zPotMin_select = zPotMin_select[index]
indexList_select = indexList_select[index]

# Print out sample
print()
print()
print('**************************************************')
print('**************************************************')
print('SAMPLE LIST: ID, MDelta(crit), RDelta(crit), [x,y,z]_MinPot')
print('ID is position in array sorted in MDelta(crit) - ID=0 is most massive cluster')
print('MDelta(crit) is SO mass in units of 1e13 Msun')
print('RDelta(crit) is SO radius in units of Mpc')
print('[x,y,z]_MinPot is 3D position of particle with minimum potential in Mpc')
for i in np.arange(numHaloes_select):
    print(indexList_select[i],
          '{:.3f}'.format(MDeltac_select[i] / 1.e13),
          '{:.3f}'.format(RDeltac_select[i]),
          '{:.3f}'.format(xPotMin_select[i]),
          '{:.3f}'.format(yPotMin_select[i]),
          '{:.3f}'.format(zPotMin_select[i]))
print('**************************************************')
print('**************************************************')
print()
print()

# Sanity check - find the closest halo that is more massive
print('Sanity check: 2nd last column (nearest massive object) should always be more than last column (minDist)')
for i in np.arange(numHaloes_select):
    dx = wrap(xPotMin - xPotMin_select[i], boxMpc)
    dy = wrap(yPotMin - yPotMin_select[i], boxMpc)
    dz = wrap(zPotMin - zPotMin_select[i], boxMpc)
    dr2 = (dx ** 2) + (dy ** 2) + (dz ** 2)
    index = np.where((MDeltac > minMassFrac * MDeltac_select[i]) & (dr2 > 0.001))[0]
    print(i, index.size, np.sqrt(dr2[index].min()), np.max([minDistMpc, minDistFac * RDeltac_select[i]]))

# Print to txt file
with open(f"{params['output_dir']}/groupnumbers_defaultSept.txt", "w") as text_file:
    print(f"# mass_sort: {'M_500crit' if massChoice else 'M_200crit'}", file=text_file)
    print("# Halo index:", file=text_file)
    for i in np.arange(numHaloes_select):
        print(f"{indexList_select[i]:d}", file=text_file)

with open(f"{params['output_dir']}/selected_halos_defaultSept.txt", "w") as text_file:
    print(f"# mass_sort: {'M_500crit' if massChoice else 'M_200crit'}", file=text_file)
    print("# Halo index, M{delta}c/1.e13 [Msun], r{delta}c [Mpc], xPotMin [Mpc], yPotMin [Mpc], zPotMin [Mpc]",
          file=text_file)
    for i in np.arange(numHaloes_select):
        print("%d, %.3f, %.3f, %.3f, %.3f, %.3f" % (
            indexList_select[i],
            MDeltac_select[i] / 1e13,
            RDeltac_select[i],
            xPotMin_select[i],
            yPotMin_select[i],
            zPotMin_select[i]
        ), file=text_file)
