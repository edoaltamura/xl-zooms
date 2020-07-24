import swiftsimio as sw
import numpy as np

print("Loading halos selected...")
lines = np.loadtxt("halo_selected.txt", comments="#", delimiter=",", unpack=False).T
print("log10(M200c / Msun): ", np.log10(lines[1]*1e13))
print("R200c: ", lines[2])
print("Centre of potential coordinates: (xC, yC, zC)")
for i in range(3):
    print(f"\tHalo {i:d}:\t{lines[3,i]:2.1f}, {lines[4,i]:2.1f}, {lines[5,i]:2.1f}")

