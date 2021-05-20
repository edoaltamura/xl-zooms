# from mpi4py import MPI
import sys
import os

sys.path.append("..")

from scaling_relations import CoolingTimes
from register import parser

parser.add_argument(
    '-s',
    '--snapshot-number',
    type=int,
    default=0
)

args = parser.parse_args()

# comm = MPI.COMM_WORLD
# num_processes = comm.size
# rank = comm.rank

name = 'L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth'
dir = '/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_birthprops/'
analysis = '/cosma/home/dp004/dc-alta2/data7/xl-zooms/analysis'

# Data assignment can be done through independent operations
for snap_number in range(args.snapshot_number, 2523):  # snap_number % num_processes == rank and
    if not os.path.isfile(analysis + f"cooling_times_{name}_{snap_number:04d}.png"):

        print(f"Rank {0} processing snapshot {snap_number}")

        s = dir + f"snapshots/{name}_{snap_number:04d}.hdf5"
        c = dir + f"stf/{name}_{snap_number:04d}/{name}_{snap_number:04d}.properties"
        gf = CoolingTimes()

        try:
            gf.process_single_halo(
                path_to_snap=s,
                path_to_catalogue=c,
                agn_time=None,
                z_agn_start=18,
                z_agn_end=0
            )
        except Exception as e:
            print(e, f"\n[!!!] Snap number {snap_number:04d} could not be processed.")
            # raise e

        del gf

# comm.Barrier()
