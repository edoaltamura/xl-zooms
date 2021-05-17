import velociraptor
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool


def smooth(data, window_width, order: int = 5):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

    # Make polynomial fit
    steps = np.arange(len(data))
    coefs = poly.polyfit(steps[window_width // 2:-window_width // 2], ma_vec[:-1], order)
    ma_vec = poly.polyval(steps, coefs)

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

r500_smoothed = smooth(r500, window)
xcminpot_smoothed = smooth(xcminpot, window)
ycminpot_smoothed = smooth(ycminpot, window)
zcminpot_smoothed = smooth(zcminpot, window)

plt.plot(r500 - r500_smoothed, label='r500')
plt.plot(xcminpot - xcminpot_smoothed, label='xcminpot')
plt.plot(ycminpot - ycminpot_smoothed, label='ycminpot')
plt.plot(zcminpot - zcminpot_smoothed, label='zcminpot')

plt.legend()
plt.show()
