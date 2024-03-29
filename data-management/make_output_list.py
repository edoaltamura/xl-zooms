import numpy as np
from astropy.units import Gyr
from astropy.cosmology import LambdaCDM, z_at_value

#######################################################################################

cosmology = {
    'h': 0.67660000,  # Reduced Hubble constant
    'Omega_m': 0.31110000,  # Matter density parameter
    'Omega_lambda': 0.68890000,  # Dark-energy density parameter
    'Omega_b': 0.04897000  # Baryon density parameter
}

# snap_redshifts = np.array([
#     18.08, 15.28, 13.06, 11.26, 9.79, 8.57, 7.54, 6.67, 5.92, 5.28, 4.72,
#     4.24, 3.81, 3.43, 3.09, 2.79, 2.52, 2.28, 2.06, 1.86, 1.68, 1.51, 1.36,
#     1.21, 1.08, 0.96, 0.85, 0.74, 0.64, 0.55, 0.46, 0.37, 0.29, 0.21, 0.14,
#     0.07, 1.e-7
# ])

snap_redshifts = np.empty(0)

redshift_start = 5.
redshift_end = 1.e-7
snip_time_delta = 5.0e-3  # Time difference between snipshots in Gyr (Yannick suggests 5 Myr)

all_snapshots = True

#######################################################################################

Planck18 = LambdaCDM(
    cosmology['h'] * 100,
    cosmology['Omega_m'],
    cosmology['Omega_lambda'],
    Ob0=cosmology['Omega_b']
)

time_start = Planck18.age(redshift_start)
time_end = Planck18.age(redshift_end)
snip_time_delta *= Gyr
number_of_snips = int((time_end - time_start) / snip_time_delta)
snip_times = np.linspace(time_start, time_end, number_of_snips, dtype=np.float)

snip_redshifts = np.zeros(len(snip_times), dtype=np.float64)
for i in range(len(snip_times)):
    snip_redshifts[i] = z_at_value(Planck18.age, snip_times[i])

if len(snap_redshifts) > 0:

    # Delete duplicate redshifts
    for z_snap in snap_redshifts:
        idx_close = np.where(np.abs(snip_redshifts - z_snap) <= snip_time_delta.value / 2)[0]
        snip_redshifts = np.delete(snip_redshifts, idx_close)

# Collect all outputs
if all_snapshots:
    out_redshift = snip_redshifts
    out_name = np.asarray(['Snapshot'] * len(out_redshift))
else:
    out_redshift = np.concatenate((snap_redshifts, snip_redshifts))
    out_name = ['Snapshot'] * len(snap_redshifts) + ['Snipshot'] * len(snip_redshifts)
    out_name = np.asarray(out_name)

# Sort and make monotonic
sort_key = np.argsort(out_redshift)[::-1]
out_redshift = out_redshift[sort_key]
out_name = out_name[sort_key]
out_redshift[-1] = 0.

# Write to file
if all_snapshots:
    with open("output_list.txt", "w") as text_file:
        print("# Redshift", file=text_file)
        for z, name in zip(out_redshift, out_name):
            print(f"{z:.8f}", file=text_file)
            print(f"{z:.8f}")
else:
    with open("output_list.txt", "w") as text_file:
        print("# Redshift, Select Output", file=text_file)
        for z, name in zip(out_redshift, out_name):
            print(f"{z:.8f}, {name:s}", file=text_file)
            print(f"{z:.8f}, {name:s}")

print(len(out_name))