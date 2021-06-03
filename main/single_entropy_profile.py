import sys

sys.path.append("..")

from scaling_relations import EntropyProfiles
from register import find_files

snap, cat = find_files()

gf = EntropyProfiles(max_radius_r500=1.5)
print(
    gf.display_single_halo(
        path_to_snap=snap,
        path_to_catalogue=cat
    )
)

# print(
#     gf.process_catalogue()
# )
