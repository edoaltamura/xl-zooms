import sys

sys.path.append("..")

from test_files import cat, snap
from scaling_relations import VRProperties

gf = VRProperties()
print(
    gf.process_single_halo(
        path_to_snap=snap,
        path_to_catalogue=cat
    )
)

print(
    gf.process_catalogue()
)
