import numpy as np
import unyt
import h5py
import swiftsimio as sw

author = "SK"
data = {}
data['parent'] = {}
data['zoom'] = {}

# PARENT DATA
lines = np.loadtxt(f"outfiles/halo_selected_{author}.txt", comments="#", delimiter=",", unpack=False).T
data['parent']['M200c'] = lines[1] * 1e13
data['parent']['r200c'] = lines[2]
data['parent']['Xcminpot'] = lines[3]
data['parent']['Ycminpot'] = lines[4]
data['parent']['Zcminpot'] = lines[5]

# ZOOM DATA
velociraptor_properties = [
    f"/cosma6/data/dp004/dc-alta2/xl-zooms/halo_{author}{i}_0001/halo_{author}{i}_0001.properties.0"
    for i in range(3)
]

M200c_zoom = []
R200c_zoom = []
x_zoom = []
y_zoom = []
z_zoom = []

for vr_path in velociraptor_properties:
    with h5py.File(vr_path, 'r') as vr_file:
        M200c_zoom.append(vr_file['/Mass_200crit'][0] * 1e10)
        R200c_zoom.append(vr_file['/R_200crit'][0])
        x_zoom.append(vr_file['/Xcminpot'][0])
        y_zoom.append(vr_file['/Ycminpot'][0])
        z_zoom.append(vr_file['/Zcminpot'][0])

data['zoom']['M200c'] = M200c_zoom
data['zoom']['r200c'] = R200c_zoom
data['zoom']['Xcminpot'] = x_zoom
data['zoom']['Ycminpot'] = y_zoom
data['zoom']['Zcminpot'] = z_zoom

print("CALCULATION DETAILS:\n\t (prop_parent - prop_zoom)/prop_parent * 100\n")

for halo_id in range(3):
    print(f"\n\nComparison halo {author} - {halo_id}")
    for key in data['parent']:
        difference = (1 - data['zoom'][key][halo_id]/data['parent'][key][halo_id])*100
        sign = '+' if difference >= 0 else ''
        warning = '!'*int(difference)
        print(f"\t{key:<10s}\t{sign}{difference:<2.3f} %\t{warning:<30s}")


