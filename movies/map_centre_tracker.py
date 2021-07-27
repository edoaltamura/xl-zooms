import velociraptor
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool

import sys
import os

sys.path.append("..")

from register import default_output_directory, find_files, xlargs, set_mnras_stylesheet


def smooth(data, window_width, order: int = 5):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

    # Make polynomial fit
    steps = np.arange(len(data))
    coefs = poly.polyfit(steps[window_width // 2:-window_width // 2], ma_vec[:-1], order)
    ma_vec = poly.polyval(steps, coefs)

    return ma_vec


def read(snapshot_number):
    _, path_to_catalogue = find_files()

    path_to_catalogue.replace(f'{xlargs.snapshot_number:04d}', f'{snapshot_number:04d}')

    vr_handle = velociraptor.load(path_to_catalogue)
    r500 = vr_handle.spherical_overdensities.r_500_rhocrit[0].to('Mpc').value
    xcminpot = vr_handle.positions.xcminpot[0].to('Mpc').value
    ycminpot = vr_handle.positions.ycminpot[0].to('Mpc').value
    zcminpot = vr_handle.positions.zcminpot[0].to('Mpc').value
    return r500, xcminpot, ycminpot, zcminpot


# Find all snapshot numbers in the directory
snaps_path = os.path.join(xlargs.run_directory, 'snapshots')
snap_numbers_from_outputs = []
for file in os.listdir(snaps_path):
    if file.endswith('.hdf5'):
        snap_numbers_from_outputs.append(int(file[-9:-5]))

catalogues_path = os.path.join(xlargs.run_directory, 'stf')
snap_numbers_from_catalogues = []
for file in os.listdir(catalogues_path):
    if file[0].isalpha():
        print(file[-5:-1])
        snap_numbers_from_catalogues.append(int(file[-5:-1]))

snap_numbers_from_outputs.sort()
snap_numbers_from_catalogues.sort()

assert snap_numbers_from_catalogues == snap_numbers_from_outputs

snap_numbers = snap_numbers_from_outputs
del snap_numbers_from_outputs, snap_numbers_from_catalogues

# Read data in parallel
with Pool() as pool:
    results = list(
        tqdm(
            pool.imap(read, snap_numbers),
            total=len(snap_numbers)
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

output = np.c_[
    xcminpot_smoothed,
    ycminpot_smoothed,
    zcminpot_smoothed,
    r500_smoothed
]
np.save(f'{os.path.basename(xlargs.run_directory)}_centre_trace.npy', output)

if not xlargs.quiet:
    set_mnras_stylesheet()
    plt.plot(r500 - r500_smoothed, label='r500')
    plt.plot(xcminpot - xcminpot_smoothed, label='xcminpot')
    plt.plot(ycminpot - ycminpot_smoothed, label='ycminpot')
    plt.plot(zcminpot - zcminpot_smoothed, label='zcminpot')
    plt.legend()
    plt.show()
