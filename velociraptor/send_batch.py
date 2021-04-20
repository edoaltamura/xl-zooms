import sys
import os

import argparse
import os.path

parser = argparse.ArgumentParser()

parser.add_argument(
    '-d',
    '--directories',
    type=str,
    nargs='+',
    required=True
)

args = parser.parse_args()

for i, run_directory in enumerate(args.directories):
    snaps_path = os.path.join(run_directory, 'snapshots')
    catalogues_path = os.path.join(run_directory, 'stf')



    if os.path.isdir(snaps_path):
        number_snapshots = len([file for file in os.listdir(snaps_path) if file.endswith('.hdf5')])
    else:
        number_snapshots = 0

    if os.path.isdir(catalogues_path):
        number_catalogues = len([subdir for subdir in os.listdir(catalogues_path)])
    else:
        number_catalogues = 0

    if (
            (number_snapshots > 0) and
            (number_catalogues > 0) and
            (number_snapshots == number_catalogues)
    ):
        self.complete_runs[i] = True