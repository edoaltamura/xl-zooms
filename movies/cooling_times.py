import sys
import os
import traceback

sys.path.append("..")

from scaling_relations import CoolingTimes
from register import find_files, xlargs

gf = CoolingTimes()

s, c = find_files()


try:
    gf.process_single_halo(
        path_to_snap=s,
        path_to_catalogue=c,
        agn_time=None
    )
except Exception as e:
    print(f"Snap number {xlargs.snapshot_number:04d} could not be processed.", e, sep='\n')
    traceback.print_exc()
