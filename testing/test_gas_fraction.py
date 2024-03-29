import sys

sys.path.append("..")

from scaling_relations import GasFractions
from register import find_files
snap, cat = find_files()

gf = GasFractions()
print(
    gf.process_single_halo(
        path_to_snap=snap,
        path_to_catalogue=cat
    )
)

print(
    gf.process_catalogue()
)
