import velociraptor
import numpy as np
import matplotlib.pyplot as plt


dir = '/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_alpha1p0/'

r500 = []
xcminpot = []
ycminpot = []
zcminpot = []

for snapshot_number in range(2523):
    path_to_catalogue = dir + (
        f"stf/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_{snapshot_number:04d}/"
        f"L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_{snapshot_number:04d}.properties"
    )

    vr_handle = velociraptor.load(path_to_catalogue)
    r500.append(vr_handle.spherical_overdensities.r_500_rhocrit[0].to('Mpc').value)
    xcminpot.append(vr_handle.positions.xcminpot[0].to('Mpc').value)
    ycminpot.append(vr_handle.positions.ycminpot[0].to('Mpc').value)
    zcminpot.append(vr_handle.positions.zcminpot[0].to('Mpc').value)

r500 = np.asarray(r500)
xcminpot = np.asarray(xcminpot)
ycminpot = np.asarray(ycminpot)
zcminpot = np.asarray(zcminpot)

plt.plot(r500 - np.mean(r500), label='r500')
plt.plot(xcminpot - np.mean(xcminpot), label='xcminpot')
plt.plot(ycminpot - np.mean(ycminpot), label='ycminpot')
plt.plot(zcminpot - np.mean(zcminpot), label='zcminpot')
plt.legend()
plt.show()