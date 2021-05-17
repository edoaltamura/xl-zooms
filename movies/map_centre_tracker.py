import velociraptor
import numpy as np
import numpy.polynomial.polynomial as poly
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
    results = list(
        tqdm(
            pool.imap(read, range(2523)),
            total=2523
        )
    )
    r500 = np.asarray(list(results)).T[0]
    xcminpot = np.asarray(list(results)).T[1]
    ycminpot = np.asarray(list(results)).T[2]
    zcminpot = np.asarray(list(results)).T[3]

window = 150
steps = np.arange(2523)
r500_smoothed = smooth(r500 - np.mean(r500), window)
xcminpot_smoothed = smooth(xcminpot, window)
ycminpot_smoothed = smooth(ycminpot, window)
zcminpot_smoothed = smooth(zcminpot, window)

coefs = poly.polyfit(steps[window // 2:-window // 2], zcminpot_smoothed, 4)
ffit = poly.polyval(steps, coefs)

# plt.plot(r500[:l] - r500_smoothed, label='r500')
# plt.plot(xcminpot[:l] - xcminpot_smoothed, label='xcminpot')
# plt.plot(ycminpot[:l] - ycminpot_smoothed, label='ycminpot')
plt.plot(zcminpot_smoothed, label='zcminpot_smoothed')
plt.plot(zcminpot[window // 2:-window // 2], label='zcminpot')

plt.legend()
plt.show()
