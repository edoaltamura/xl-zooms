import swiftsimio as sw
import numpy as np

lines = np.loadtxt("halo_selected.txt", comments="#", delimiter=",", unpack=False)
print(lines)
