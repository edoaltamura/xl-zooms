import sys

sys.path.append("..")

from scaling_relations import XrayLuminosities
from test_files import cat, snap

gf = XrayLuminosities()
print(
    gf.process_single_halo(
        path_to_snap=snap,
        path_to_catalogue=cat
    )
)

print(
    gf.process_catalogue()
)