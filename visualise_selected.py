import swiftsimio as sw
import numpy as np

lines = np.loadtxt("halo_selected.txt", comments="#", delimiter=",", unpack=False).T
print("log10(M200c / Msun): ", np.log10(lines[1]*1e13))
print("R200c: ", lines[2])
print("Centre of Potential cordinates: (xC, yC, zC)")
for i in range(3):
    print(f"\t{lines[3,i]}, {lines[4,i]}, {lines[5,i]}")

