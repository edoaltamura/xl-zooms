import sys
import os

sys.path.append("..")

from scaling_relations import CoolingTimes
from register import parser

gf = CoolingTimes()

parser.add_argument(
    '-s',
    '--snapshot-number',
    type=int,
    required=True
)

parser.add_argument(
    '-b',
    '--run-directory',
    type=int,
    required=True
)

args = parser.parse_args()

s = ''
for file in os.listdir(os.path.join(args.run_directory, 'snapshots')):
    if file.endswith(f"_{args.snapshot_number:04d}.hdf5"):
        s = os.path.join(args.run_directory, 'snapshots', file)
        break

c = ''
for subdir in os.listdir(os.path.join(args.run_directory, 'stf')):
    if subdir.endswith(f"_{args.snapshot_number:04d}"):
        for file in os.listdir(os.path.join(args.run_directory, 'stf', subdir)):
            if file.endswith(f"_{args.snapshot_number:04d}.properties"):
                c = os.path.join(args.run_directory, 'stf', subdir, file)
                break

assert s and c

try:
    gf.process_single_halo(
        path_to_snap=s,
        path_to_catalogue=c,
        agn_time=None
    )
except Exception as e:
    print(f"Snap number {args.snapshot_number:04d} could not be processed.", e, sep='\n')
