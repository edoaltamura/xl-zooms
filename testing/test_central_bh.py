import sys

sys.path.append("..")

from scaling_relations import CentralBH
from test_files import cat, snap

gf = CentralBH()
print(
    gf.process_single_halo(
        path_to_snap=snap,
        path_to_catalogue=cat
    )
)
