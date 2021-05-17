import velociraptor
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool


def smooth(data, window_width):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec


def read(snapshot_number):
    path_to_catalogue = dir + (
        f"stf/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_{snapshot_number:04d}/"
        f"L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_{snapshot_number:04d}.properties"
    )

    vr_handle = velociraptor.load(path_to_catalogue)
    r500 = vr_handle.spherical_overdensities.r_500_rhocrit[0].to('Mpc').value
    xcminpot = vr_handle.positions.xcminpot[0].to('Mpc').value
    ycminpot = vr_handle.positions.ycminpot[0].to('Mpc').value
    zcminpot = vr_handle.positions.zcminpot[0].to('Mpc').value
    return r500, xcminpot, ycminpot, zcminpot


dir = '/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_alpha1p0/'


with Pool() as pool:
    results = list(tqdm(
        pool.imap(read, range(2523)),
        total=2523
    ))
    r500 = np.asarray(list(results)).T[0]
    xcminpot = np.asarray(list(results)).T[1]
    ycminpot = np.asarray(list(results)).T[2]
    zcminpot = np.asarray(list(results)).T[3]


plt.plot(r500 - np.mean(r500), label='r500')
plt.plot(xcminpot - np.mean(xcminpot), label='xcminpot')
plt.plot(ycminpot - np.mean(ycminpot), label='ycminpot')
plt.plot(zcminpot - np.mean(zcminpot), label='zcminpot')

plt.plot(smooth(xcminpot - np.mean(xcminpot), 3), label='xcminpot_smooth3')
plt.plot(smooth(ycminpot - np.mean(ycminpot), 3), label='ycminpot_smooth3')
plt.plot(smooth(zcminpot - np.mean(zcminpot), 3), label='zcminpot_smooth3')

plt.plot(smooth(xcminpot - np.mean(xcminpot), 7), label='xcminpot_smooth7')
plt.plot(smooth(ycminpot - np.mean(ycminpot), 7), label='ycminpot_smooth7')
plt.plot(smooth(zcminpot - np.mean(zcminpot), 7), label='zcminpot_smooth7')

plt.plot(smooth(xcminpot - np.mean(xcminpot), 15), label='xcminpot_smooth15')
plt.plot(smooth(ycminpot - np.mean(ycminpot), 15), label='ycminpot_smooth15')
plt.plot(smooth(zcminpot - np.mean(zcminpot), 15), label='zcminpot_smooth15')

plt.legend()
plt.show()
