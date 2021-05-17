import velociraptor
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from scipy.interpolate import interp1d
from scipy import arange, array, exp


def smooth(data, window_width):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec


def read(snapshot_number):
    path_to_catalogue = dir + (
        f"stf/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_{snapshot_number:04d}/"
        f"L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_{snapshot_number:04d}.properties"
    )

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
        elif x > xs[-1]:
            return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(list(map(pointwise, array(xs))))

    return ufunclike

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

f_i = interp1d(steps[window // 2:-window // 2], xcminpot_smoothed)
f_x = extrap1d(f_i)

zcminpot_smoothed = np.r_[
    f_x(steps[:window // 2]),
    zcminpot_smoothed,
    f_x(steps[-window // 2:]),
]

# plt.plot(r500[:l] - r500_smoothed, label='r500')
# plt.plot(xcminpot[:l] - xcminpot_smoothed, label='xcminpot')
# plt.plot(ycminpot[:l] - ycminpot_smoothed, label='ycminpot')
plt.plot(zcminpot_smoothed, label='zcminpot_smoothed')
plt.plot(zcminpot, label='zcminpot')

plt.legend()
plt.show()
