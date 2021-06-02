import sys

sys.path.append("..")

from scaling_relations import EntropyProfiles
from test_files import cat, snap

gf = EntropyProfiles()
print(
    gf.process_single_halo(
        path_to_snap=snap,
        path_to_catalogue=cat
    )
)

# print(
#     gf.process_catalogue()
# )
